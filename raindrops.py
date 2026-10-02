"""
raindrops.py (v4)
Mo phong giot mua tren kinh - phien ban sua loi "bong bong lo lung":
hinh dang bat quy tac, chiu luc trong truong, phan quang sac net, khong con
vien sang mo lan rong kieu vien tron hoan hao.

Thay doi so voi v3 (nguyen nhan gay cam giac "bong bong noi"):
1. HINH DANG BAT QUY TAC: bien dang giot khong con la hinh elip/tron hoan hao.
   Ban kinh vien duoc nhieu (perturb) theo goc bang tong vai ham sin/cos ngau
   nhien (nhu ve tay), + 1 "buou" phinh o day giot do trong luc keo nuoc xuong
   -> giot tron dung yen cung khong con tron tuyet doi, giong giot nuoc dong
   that tren kinh hon.
2. VIEN FRESNEL MONG & SAC: truoc day fresnel lan dan tu tam ra ria tao thanh
   1 quang sang lon nhin nhu vien bong bong. Gio fresnel chi kich hoat that
   manh trong 1 DAI MONG sat ria giot (dung edge-mask cat gon), phan than giot
   giu trong suot/khuc xa binh thuong.
3. PHAN QUANG (specular) SAC NET KIEU "CATCH-LIGHT": ngoai spec Blinn-Phong
   voi so mu rat cao (spot nho), them 1 diem sang phu dang tron nho, cuc net
   (nhu anh phan chieu bau troi/den duong that tren mat nuoc), vi tri lech ve
   phia tren-nghieng cho tu nhien, KHONG con la vung gauss to mo.
4. VET CHAN/DAY GIOT: them 1 vanh toi rat mong ngay sat mep duoi giot (noi
   nuoc "dinh" vao kinh that su) de "neo" giot xuong be mat, tranh cam giac
   giot troi lo lung trong khong gian.
5. GIOT CHAY DAI HON, NGOAC NGOEO RO HON, DUOI THON NHON O DAU TREN VA
   PHINH TO METHOD LECH O DAU DUOI (giong giot bi trong luc/gio keo that).

Yeu cau: opencv-python, numpy
"""

import cv2
import numpy as np

# Mau "bau troi/anh sang moi truong" xap xi, dung khi Fresnel phan xa manh o vien giot (BGR)
_SKY_AMBIENT = np.array([225.0, 223.0, 219.0])
_WATER_ETA = 1.0 / 1.33  # khong khi -> nuoc


# Vung KHONG phai kinh chan gio (toa do 0-1): vat toi o goc duoi ben trai anh dashcam nay
_DEFAULT_EXCLUDE = [[(0.0, 0.69), (0.05, 0.72), (0.085, 0.86), (0.095, 0.95), (0.0, 0.95)]]


def _irregular_radius(theta, rng, n_harm=2, amp=0.08, gravity_bulge=0.0):
    """
    Ban kinh vien giot (chuan hoa quanh 1.0) theo goc theta, hoi meo tu nhien
    bang tong 1-2 hai hoa TAN SO THAP (khong tao hinh sao/hoa) + 1 buou phinh
    o phia duoi do trong luc.
    """
    r = np.ones_like(theta)
    for _ in range(n_harm):
        k = rng.integers(2, 4)  # tan so thap -> bien dang meo nhe, khong thanh canh hoa
        phase = rng.uniform(0, 2 * np.pi)
        a = rng.uniform(0.015, amp)
        r = r + a * np.cos(k * theta + phase)
    if gravity_bulge > 0:
        down = np.clip(np.sin(theta), 0, 1)
        r = r + gravity_bulge * (down ** 1.5)
    return np.clip(r, 0.75, None)


def _render_drop(result, base_sharp, h, w, cx, cy, drop_w, drop_h,
                  alpha_scale, light_dir, rng, gravity_bulge=0.22, irregular=0.13,
                  highlight_scale=1.0, glass=None):
    """Ve 1 giot bat quy tac tai (cx, cy) len anh `result`, lay mau tu `base_sharp`."""
    pad = 1.35  # mo rong bbox de chua phan buou/meo vuot ra ngoai elip goc
    x0, x1 = max(int(cx - drop_w * pad), 0), min(int(cx + drop_w * pad) + 1, w)
    y0, y1 = max(int(cy - drop_h * pad), 0), min(int(cy + drop_h * pad) + 1, h)
    pw, ph = x1 - x0, y1 - y0
    if pw < 3 or ph < 3:
        return

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    nx = (xx - cx) / drop_w
    ny = (yy - cy) / drop_h
    dist = np.sqrt(nx ** 2 + ny ** 2) + 1e-6
    theta = np.arctan2(ny, nx)

    r_limit = _irregular_radius(theta, rng, amp=irregular, gravity_bulge=gravity_bulge)
    dist_norm = dist / r_limit
    inside = dist_norm <= 1.0
    if not inside.any():
        return
    dist_norm_c = np.clip(dist_norm, 0, 1)

    # phap tuyen 3D uoc luong tu dist_norm (mai vom cuc bo theo bien khong deu)
    nz = np.sqrt(np.clip(1.0 - dist_norm_c ** 2, 0.0, 1.0))
    inv_r = 1.0 / np.maximum(dist, 1e-6)
    ux, uy = nx * inv_r, ny * inv_r  # huong xuyen tam, chuan hoa
    sin_comp = np.sqrt(np.clip(1 - nz ** 2, 0, 1))
    dnx, dny = ux * sin_comp, uy * sin_comp
    cosI = nz

    # --- Khuc xa theo dinh luat Snell ---
    sin2t = np.clip((_WATER_ETA ** 2) * (1.0 - cosI ** 2), 0.0, 1.0)
    cosT = np.sqrt(1.0 - sin2t)
    k = _WATER_ETA * cosI - cosT
    depth_scale = 3.4 * min(drop_w, drop_h)
    map_x = np.clip(xx + k * dnx * depth_scale, 0, w - 1).astype(np.float32)
    map_y = np.clip(yy + k * dny * depth_scale, 0, h - 1).astype(np.float32)
    patch = cv2.remap(
        base_sharp, map_x, map_y,
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    ).astype(np.float32)

    # --- Fresnel: chi la 1 DAI MONG sat ria giot, khong lan vao than giot ---
    R0 = 0.02
    fresnel_raw = R0 + (1 - R0) * np.clip(1 - cosI, 0, 1) ** 4.5
    edge_mask = np.clip((dist_norm_c - 0.78) / 0.22, 0, 1) ** 2.2  # chi kich hoat gan mep, mem hon
    fresnel = np.clip(fresnel_raw * edge_mask * 1.4, 0, 0.85)
    patch = patch * (1 - fresnel)[..., None] + _SKY_AMBIENT[None, None, :] * fresnel[..., None]

    # --- Phan quang: CHI la cac diem sang nho, sac net (khong con vet sang mo
    #     kieu "hat tuyet"). Duoc cong SAU khi tron alpha de khong bi nhoe,
    #     va giam theo kich thuoc giot (giot nho -> diem sang cuc nho, mo hon) ---
    lx, ly, lz = light_dir
    ndotl = np.clip(dnx * lx + dny * ly + nz * lz, 0, 1)
    shininess = rng.uniform(160, 320)
    spec_amp = rng.uniform(170, 230)
    hl_x = rng.uniform(-0.35, 0.35)
    hl_y = rng.uniform(-0.55, -0.15)
    sigma = rng.uniform(0.05, 0.09)
    catch_amp = rng.uniform(120, 190)

    size_f = float(np.clip(min(drop_w, drop_h) / 20.0, 0.35, 1.0))
    hl = (ndotl ** shininess) * spec_amp * 0.6
    hx_px, hy_px = cx + hl_x * drop_w, cy + hl_y * drop_h
    sig_px = 0.7 + sigma * 6.0                     # ~1.0-1.25 px, khong phu thuoc kich thuoc giot
    catch = np.exp(-((xx - hx_px) ** 2 + (yy - hy_px) ** 2) / (2 * sig_px ** 2)) * inside
    hl = (hl + catch * catch_amp * 0.6) * highlight_scale * size_f

    # --- Vanh toi mong sat mep duoi cung: "neo" giot vao be mat kinh ---
    down_w = np.clip(np.sin(theta), 0, 1)
    contact = (np.clip((dist_norm_c - 0.86) / 0.14, 0, 1) ** 2) * (0.35 + 0.65 * down_w)
    patch *= (1.0 - 0.30 * contact)[..., None]

    # --- Khoi giot: hoi toi dan ra vien (rat nhe, de khong tao vong tron ro) ---
    body_shadow = np.clip((dist_norm_c - 0.55) / 0.45, 0, 1) ** 2
    patch *= (1.0 - 0.10 * body_shadow)[..., None]

    # --- Alpha: giu trong suot o giua, mep giam nhanh (khong con quang mo rong) ---
    alpha = np.clip(1.0 - dist_norm_c ** 2, 0, 1.4) ** 1.4
    alpha *= alpha_scale
    alpha = np.where(inside, alpha, 0.0)
    g = glass[y0:y1, x0:x1] if glass is not None else 1.0   # 0 = khong phai kinh (taplo...)
    alpha = (alpha * g)[..., None]

    region = result[y0:y1, x0:x1]
    blended = region * (1 - alpha) + np.clip(patch, 0, 255) * alpha

    blended += (hl * inside * g)[..., None]
    result[y0:y1, x0:x1] = np.clip(blended, 0, 255)


def _apply_rivulets(result, base_sharp, h, w, rng, light_dir,
                    num=26, y_limit_frac=0.90, drift=0.15, glass=None):
    """
    Ve cac DONG NUOC CHAY XUONG tren kinh chan gio (nhin tu ghe lai):
    - moi dong la 1 duong ngoac ngoeo, do rong thay doi, co cac "bua" phinh
      nuoc dong lai roi thon dan (giong nuoc chay tren kinh that)
    - dong nuoc duoc dung thanh HEIGHT FIELD -> phap tuyen -> khuc xa (dao anh
      nhu thau kinh), phan quang o suon sang, toi o suon doi dien
    - dau duoi moi dong co 1 giot tron phinh (giot dang truot xuong)
    - hoi tan ra ngoai theo chieu gio khi xe chay (drift), khong xuong duong ca
      vung taplo o day anh (y_limit_frac)
    """
    y_lim = h * y_limit_frac
    mask = np.zeros((h, w), np.uint8)
    heads = []

    xs0 = (np.arange(num) + rng.random(num)) * (w / num)
    rng.shuffle(xs0)
    SH = 4  # sub-pixel shift cho cv2.line (muot hon)

    for x0 in xs0:
        y0 = rng.uniform(-0.05 * h, 0.55 * h)
        y_end = min(y0 + rng.uniform(0.15 * h, 0.55 * h), y_lim)
        if y_end - y0 < 20:
            continue
        base_w = rng.uniform(1.8, 4.0)
        ys = np.arange(y0, y_end, 2.0)
        n = len(ys)
        t = np.linspace(0, 1, n)

        # duong di ngoac ngoeo + troi nhe ra ngoai (huong tu tam ra ria)
        bias = ((x0 - w / 2) / (w / 2)) * drift
        px = np.empty(n)
        x, vx = x0, 0.0
        for i in range(n):
            vx = 0.88 * vx + rng.normal(0, 0.28)
            x += vx + bias
            px[i] = x

        # do rong: thon o dau tren, dao dong nhe, them cac bua phinh
        wid = base_w * (0.5 + 0.5 * np.clip(t * 4, 0, 1))
        wid *= 1 + 0.25 * np.sin(t * rng.uniform(4, 12) + rng.uniform(0, 2 * np.pi))
        for _ in range(rng.integers(0, 4)):
            tb = rng.uniform(0.15, 1.0)
            wid = wid + base_w * rng.uniform(0.8, 2.2) * np.exp(-((t - tb) / 0.035) ** 2)

        for i in range(n - 1):
            th = max(1, int(round((wid[i] + wid[i + 1]) / 2)))
            p1 = (int(round(px[i] * (1 << SH))), int(round(ys[i] * (1 << SH))))
            p2 = (int(round(px[i + 1] * (1 << SH))), int(round(ys[i + 1] * (1 << SH))))
            cv2.line(mask, p1, p2, 255, th, cv2.LINE_AA, SH)
        heads.append((px[-1], ys[-1], max(base_w * 1.9, 4.0)))

    m = mask.astype(np.float32) / 255.0
    H = cv2.GaussianBlur(m, (0, 0), 1.8)          # "chieu cao" dong nuoc
    coverage = np.clip(cv2.GaussianBlur(m, (0, 0), 0.7) * 1.3, 0, 1)
    if glass is not None:
        coverage = coverage * glass
    gy, gx = np.gradient(H)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    k = 55.0
    map_x = np.clip(xx + gx * k, 0, w - 1).astype(np.float32)
    map_y = np.clip(yy + gy * k, 0, h - 1).astype(np.float32)
    patch = cv2.remap(base_sharp, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REPLICATE).astype(np.float32)

    lx, ly, lz = light_dir
    face = (-gx * lx - gy * ly) * 9.0             # >0: suon quay ve phia anh sang
    pos = np.clip(face, 0, 1)
    neg = np.clip(-face, 0, 1)
    nxy = np.sqrt(gx ** 2 + gy ** 2) * 9.0
    n_len = np.sqrt((gx * 9.0) ** 2 + (gy * 9.0) ** 2 + 1.0)
    ndotl = np.clip((-gx * 9.0 * lx - gy * 9.0 * ly + lz) / n_len, 0, 1)
    spec = ndotl ** 60

    patch += (pos ** 1.5 * 70 + spec * 120)[..., None]
    patch *= (1.0 - 0.32 * np.clip(neg + 0.25 * nxy, 0, 1))[..., None]
    patch *= (1.0 - 0.04 * coverage)[..., None]

    a = (coverage * 0.9)[..., None]
    result[:] = result * (1 - a) + np.clip(patch, 0, 255) * a

    # giot phinh o dau duoi moi dong (dang truot xuong)
    for hx, hy, hw in heads:
        _render_drop(result, base_sharp, h, w, hx, hy, hw, hw * 1.3,
                     0.78, light_dir, rng, gravity_bulge=0.25, irregular=0.06,
                     highlight_scale=0.6, glass=glass)


def _stratified_points(n, w, h, rng):
    """Rai n diem phan bo deu tren anh (luoi co jitter) de khong bi mang trong."""
    cell = np.sqrt(w * h / n)
    cols = max(int(np.ceil(w / cell)), 1)
    rows = max(int(np.ceil(h / cell)), 1)
    cw, ch = w / cols, h / rows
    pts = []
    for r_ in range(rows):
        for c_ in range(cols):
            pts.append((int((c_ + rng.random()) * cw), int((r_ + rng.random()) * ch)))
    rng.shuffle(pts)
    return pts[:n]


def apply_raindrops(
    image,
    num_drops=200,
    min_radius=8,
    max_radius=24,
    streak_prob=0.4,
    big_prob=0.12,
    big_min_radius=28,
    big_max_radius=46,
    num_rivulets=26,
    y_limit_frac=0.89,
    exclude_polys=_DEFAULT_EXCLUDE,
    seed=None,
):
    """
    Ap hieu ung giot mua len 1 anh: hinh dang bat quy tac, khuc xa that, fresnel
    dai mong, specular sac net, co vanh chan giot -> tranh cam giac "bong bong noi".

    image : numpy.ndarray, anh BGR (nhu doc bang cv2.imread)
    num_drops : so luong giot (giot chay dai tinh la 1 du ve nhieu doan)
    min_radius / max_radius : ban kinh "dau giot" (pixel)
    streak_prob : xac suat 1 giot la dang chay dai xuong duoi (trong luc keo)
    num_rivulets : so dong nuoc chay xuong tren kinh (0 = tat)
    y_limit_frac : khong ve giot/dong nuoc duoi muc nay (taplo, mui xe)
    exclude_polys : cac da giac (toa do 0-1) khong phai kinh, khong ve giot len do
    seed : de tai lap ket qua (None = moi lan mot khac)

    return : numpy.ndarray anh BGR da co hieu ung giot mua
    """
    rng = np.random.default_rng(seed)
    h, w = image.shape[:2]
    result = image.astype(np.float32)
    base_sharp = image.astype(np.float32)

    light_dir = np.array([-0.45, -0.55, 0.70])
    light_dir = light_dir / np.linalg.norm(light_dir)

    # mat na "day la kinh": loai bo taplo/mui xe o day anh + vat the toi o goc trai
    glass = np.ones((h, w), np.float32)
    glass[int(h * y_limit_frac):, :] = 0
    for poly in exclude_polys:
        cv2.fillPoly(glass, [np.array([(int(px * w), int(py * h)) for px, py in poly], np.int32)], 0)
    glass = cv2.GaussianBlur(glass, (0, 0), 3)

    # dong nuoc chay xuong (rng rieng de khong doi bo cuc cac giot ben duoi)
    if num_rivulets > 0:
        rng_r = np.random.default_rng(None if seed is None else seed + 1)
        _apply_rivulets(result, base_sharp, h, w, rng_r, light_dir, num=num_rivulets, glass=glass)

    points = _stratified_points(num_drops, w, h, rng)

    for pt_x, pt_y in points:
        is_big = rng.random() < big_prob
        if is_big:
            r = rng.integers(big_min_radius, big_max_radius)
        else:
            r = rng.integers(min_radius, max_radius)

        # giot to it khi chay dai (tranh vet chay qua dai), giot nho chay theo streak_prob
        is_streak = rng.random() < (0.2 if is_big else streak_prob)
        if is_big and is_streak:
            r = min(r, 32)

        if is_streak:
            # --- Giot dang chay: duong ngoac ngoeo ro + dau giot phinh lech duoi ---
            head_x = pt_x
            head_y = max(pt_y, int(h * 0.12))
            trail_len = rng.integers(3, 7)
            step = rng.uniform(r * 1.8, r * 2.8)

            xs = [head_x]
            for i in range(trail_len):
                xs.append(xs[-1] + rng.normal(0, r * 0.30))
            xs = xs[::-1]
            ys = [head_y - step * (trail_len - i) for i in range(trail_len + 1)]

            alpha_scale = rng.uniform(0.55, 0.80)
            for i, (px, py) in enumerate(zip(xs, ys)):
                t = i / trail_len  # 0 = dau tren (nho, mo dan), 1 = dau giot duoi
                seg_r = r * (0.16 + 0.95 * t ** 1.7)
                seg_w = max(seg_r * (0.45 if t < 0.97 else 1.05), 1.8)
                seg_h = max(seg_r * (0.62 if t < 0.97 else 1.05), 1.8)
                seg_alpha = alpha_scale * (0.30 + 0.70 * t)
                seg_gravity = 0.30 if t > 0.9 else 0.10
                _render_drop(result, base_sharp, h, w, px, py, seg_w, seg_h,
                             seg_alpha, light_dir, rng,
                             gravity_bulge=seg_gravity, irregular=0.09,
                             highlight_scale=0.25 + 0.75 * t ** 1.5, glass=glass)
        else:
            # --- Giot tron dong yen, van hoi meo + phinh day do trong luc ---
            cx = pt_x
            cy = pt_y
            drop_w = r * rng.uniform(0.85, 1.1)
            drop_h = r * rng.uniform(0.9, 1.25)
            alpha_scale = rng.uniform(0.62, 0.85)
            _render_drop(result, base_sharp, h, w, cx, cy, drop_w, drop_h,
                         alpha_scale, light_dir, rng,
                         gravity_bulge=rng.uniform(0.12, 0.24), irregular=0.07, glass=glass)

    return np.clip(result, 0, 255).astype(np.uint8)


def simulate_raindrops(input_path, output_path, num_drops=200, seed=None):
    """Doc anh tu input_path, ap hieu ung giot mua, luu ra output_path."""
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Khong doc duoc anh: {input_path}")
    out = apply_raindrops(image, num_drops=num_drops, seed=seed)
    cv2.imwrite(output_path, out)
    return output_path


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "input.jpg"
    dst = sys.argv[2] if len(sys.argv) > 2 else "output_rain.jpg"
    simulate_raindrops(src, dst, num_drops=200, seed=42)
    print(f"Da luu: {dst}")
