# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anonymous Authors

#!/usr/bin/env python3
"""ARC operator library: direct function-passing pattern.

gen_cands(task) -> list of (name, callable) pairs
certify(task, fn) -> (passed, checked_cells, mismatches, unsupported_pairs)

Each callable takes (flat_grid, width, height) -> (new_flat_grid, nw, nh).
"""

import json
import itertools
from collections import deque

# ── grid utilities ──────────────────────────────────────────────────────────

def grid_copy(g, w, h):
    return list(g), w, h

def grid_init(w, h):
    return [0] * (w * h), w, h

def grid_get(g, w, _h, x, y):
    return g[y * w + x]

def grid_set(g, w, _h, x, y, v):
    g[y * w + x] = v

def _to_flat(grid_2d):
    h = len(grid_2d)
    w = len(grid_2d[0]) if h > 0 else 0
    return [v for row in grid_2d for v in row], w, h

def _to_2d(flat, w, h):
    return [[flat[y * w + x] for x in range(w)] for y in range(h)]

# ── bbox ────────────────────────────────────────────────────────────────────

def find_nonzero_bbox(g, w, h, background=0):
    min_x, min_y, max_x, max_y = w, h, -1, -1
    for y in range(h):
        for x in range(w):
            if g[y * w + x] != background:
                if x < min_x: min_x = x
                if y < min_y: min_y = y
                if x > max_x: max_x = x
                if y > max_y: max_y = y
    if min_x > max_x:
        return None
    return min_x, min_y, max_x, max_y

# ── connected components ────────────────────────────────────────────────────

def largest_component(g, w, h, background=0):
    visited = [False] * (w * h)
    best = None
    for y in range(h):
        for x in range(w):
            idx = y * w + x
            if visited[idx] or g[idx] == background:
                continue
            comp = []
            q = deque()
            q.append((x, y))
            visited[idx] = True
            while q:
                cx, cy = q.popleft()
                comp.append((cx, cy))
                for nx, ny in [(cx-1,cy), (cx+1,cy), (cx,cy-1), (cx,cy+1)]:
                    if 0 <= nx < w and 0 <= ny < h:
                        nidx = ny * w + nx
                        if not visited[nidx] and g[nidx] != background:
                            visited[nidx] = True
                            q.append((nx, ny))
            if best is None or len(comp) > len(best):
                best = comp
    return best

# ── color map extraction ────────────────────────────────────────────────────

def extract_color_maps(task):
    maps = []
    for pair in task.get('train', []):
        inp = pair['input']
        out = pair['output']
        ih, iw = len(inp), len(inp[0]) if inp else 0
        oh, ow = len(out), len(out[0]) if out else 0
        inp_flat = [v for row in inp for v in row]
        out_flat = [v for row in out for v in row]
        cmap = {}
        for y in range(min(ih, oh)):
            for x in range(min(iw, ow)):
                ic = inp_flat[y * iw + x]
                oc = out_flat[y * ow + x]
                if ic not in cmap:
                    cmap[ic] = oc
        maps.append(cmap)
    return maps

def extract_cross_pair_color_map(task):
    cmap = {}
    for pair in task.get('train', []):
        inp = pair['input']
        out = pair['output']
        ih, iw = len(inp), len(inp[0]) if inp else 0
        oh, ow = len(out), len(out[0]) if out else 0
        if iw != ow or ih != oh:
            return None
        inp_flat = [v for row in inp for v in row]
        out_flat = [v for row in out for v in row]
        for y in range(ih):
            for x in range(iw):
                ic = inp_flat[y * iw + x]
                oc = out_flat[y * ow + x]
                if ic in cmap and cmap[ic] != oc:
                    return None
                cmap[ic] = oc
    return cmap if cmap else None

# ── task I/O ────────────────────────────────────────────────────────────────

def load_task(task_path):
    with open(task_path) as f:
        return json.load(f)

# ── operator factories ──────────────────────────────────────────────────────

def op_rotate90(g, w, h):
    out, _ = grid_init(h, w)
    for y in range(h):
        for x in range(w):
            grid_set(out, h, w, h - 1 - y, x, grid_get(g, w, h, x, y))
    return out, h, w

def op_rotate180(g, w, h):
    out, _, _ = grid_init(w, h)
    for y in range(h):
        for x in range(w):
            grid_set(out, w, h, w - 1 - x, h - 1 - y, grid_get(g, w, h, x, y))
    return out, w, h

def op_rotate270(g, w, h):
    out, _ = grid_init(h, w)
    for y in range(h):
        for x in range(w):
            grid_set(out, h, w, y, w - 1 - x, grid_get(g, w, h, x, y))
    return out, h, w

def op_flip_horizontal(g, w, h):
    out, _, _ = grid_init(w, h)
    for y in range(h):
        for x in range(w):
            grid_set(out, w, h, w - 1 - x, y, grid_get(g, w, h, x, y))
    return out, w, h

def op_flip_vertical(g, w, h):
    out, _, _ = grid_init(w, h)
    for y in range(h):
        for x in range(w):
            grid_set(out, w, h, x, h - 1 - y, grid_get(g, w, h, x, y))
    return out, w, h

def op_transpose(g, w, h):
    out, _ = grid_init(h, w)
    for y in range(h):
        for x in range(w):
            grid_set(out, h, w, y, x, grid_get(g, w, h, x, y))
    return out, h, w

def make_scale(factor):
    def f(g, w, h):
        nw, nh = w * factor, h * factor
        out, _, _ = grid_init(nw, nh)
        for y in range(nh):
            for x in range(nw):
                out[y * nw + x] = g[(y // factor) * w + (x // factor)]
        return out, nw, nh
    return f

def make_translate(dx, dy):
    def f(g, w, h):
        out, _, _ = grid_init(w, h)
        for i in range(w * h):
            out[i] = g[i]
        for y in range(h):
            for x in range(w):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    out[ny * w + nx] = g[y * w + x]
                else:
                    out[y * w + x] = 0
        return out, w, h
    return f

def make_translate_recolor(dx, dy, in_color, out_color):
    def f(g, w, h):
        out, _, _ = grid_init(w, h)
        for i in range(w * h):
            out[i] = g[i]
        for y in range(h):
            for x in range(w):
                if g[y * w + x] == in_color:
                    out[y * w + x] = 0
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        out[ny * w + nx] = out_color
        return out, w, h
    return f

def make_crop_background(bg):
    def f(g, w, h):
        bb = find_nonzero_bbox(g, w, h, bg)
        if bb is None:
            return grid_init(1, 1)
        mx, my, Mx, My = bb
        nw, nh = Mx - mx + 1, My - my + 1
        out, _, _ = grid_init(nw, nh)
        for y in range(nh):
            for x in range(nw):
                out[y * nw + x] = g[(my + y) * w + mx + x]
        return out, nw, nh
    return f

def make_crop_background_recolor(bg, color):
    def f(g, w, h):
        bb = find_nonzero_bbox(g, w, h, bg)
        if bb is None:
            return grid_init(1, 1)
        mx, my, Mx, My = bb
        nw, nh = Mx - mx + 1, My - my + 1
        out, _, _ = grid_init(nw, nh)
        for y in range(nh):
            for x in range(nw):
                v = g[(my + y) * w + mx + x]
                out[y * nw + x] = color if v != bg else 0
        return out, nw, nh
    return f

def make_bbox_panel_selection(out_w, out_h):
    def f(g, w, h):
        bb = find_nonzero_bbox(g, w, h, 0)
        if bb is None:
            return grid_init(out_w, out_h)
        mx, my, _, _ = bb
        out, _, _ = grid_init(out_w, out_h)
        for y in range(out_h):
            for x in range(out_w):
                sx, sy = mx + x, my + y
                if 0 <= sx < w and 0 <= sy < h:
                    out[y * out_w + x] = g[sy * w + sx]
        return out, out_w, out_h
    return f

def make_panel_selection(panel_w):
    def f(g, w, h):
        if panel_w >= w:
            return grid_init(1, 1)
        nw = w - panel_w
        out, _, _ = grid_init(nw, h)
        for y in range(h):
            for x in range(nw):
                out[y * nw + x] = g[y * w + panel_w + x]
        return out, nw, h
    return f

def make_color_map(cmap):
    def f(g, w, h):
        out, _, _ = grid_init(w, h)
        for i in range(w * h):
            out[i] = cmap.get(g[i], g[i])
        return out, w, h
    return f

def make_foreground_recolor(bg, color):
    def f(g, w, h):
        out, _, _ = grid_init(w, h)
        for i in range(w * h):
            out[i] = color if g[i] != bg else 0
        return out, w, h
    return f

def make_bbox_outline(bg, color):
    def f(g, w, h):
        bb = find_nonzero_bbox(g, w, h, bg)
        if bb is None:
            return grid_init(1, 1)
        mx, my, Mx, My = bb
        out, _, _ = grid_init(w, h)
        for i in range(w * h):
            out[i] = g[i]
        for x in range(mx, Mx + 1):
            grid_set(out, w, h, x, my, color)
            grid_set(out, w, h, x, My, color)
        for y in range(my, My + 1):
            grid_set(out, w, h, mx, y, color)
            grid_set(out, w, h, Mx, y, color)
        return out, w, h
    return f

def make_bbox_fill(bg, color):
    def f(g, w, h):
        bb = find_nonzero_bbox(g, w, h, bg)
        if bb is None:
            return grid_init(1, 1)
        mx, my, Mx, My = bb
        out, _, _ = grid_init(w, h)
        for i in range(w * h):
            out[i] = g[i]
        for y in range(my, My + 1):
            for x in range(mx, Mx + 1):
                grid_set(out, w, h, x, y, color)
        return out, w, h
    return f

# ── layer / half-grid operations ────────────────────────────────────────────

def _get_halves(g, w, h):
    mid = w // 2
    left, lw, lh = grid_init(mid, h)
    right, rw, rh = grid_init(mid, h)
    for y in range(h):
        for x in range(mid):
            left[y * mid + x] = g[y * w + x]
            right[y * mid + x] = g[y * w + mid + x]
    return (left, lw, lh), (right, rw, rh)

def make_layer_op(op_name, color):
    def f(g, w, h):
        w2 = w // 2
        if w2 == 0:
            return grid_init(1, 1)
        (left, lw, lh), (right, rw, rh) = _get_halves(g, w, h)
        nw = min(lw, rw)
        nh = min(lh, rh)
        out, _, _ = grid_init(nw, nh)
        for y in range(nh):
            for x in range(nw):
                lv = grid_get(left, lw, lh, x, y) != 0
                rv = grid_get(right, rw, rh, x, y) != 0
                if op_name == 'overlap':
                    out[y * nw + x] = color if (lv and rv) else 0
                elif op_name == 'union':
                    out[y * nw + x] = color if (lv or rv) else 0
                elif op_name == 'xor':
                    out[y * nw + x] = color if (lv != rv) else 0
                elif op_name == 'left_only':
                    out[y * nw + x] = color if (lv and not rv) else 0
                elif op_name == 'right_only':
                    out[y * nw + x] = color if (rv and not lv) else 0
        return out, nw, nh
    return f

def make_object_select_overlap(color):
    def f(g, w, h):
        mid = w // 2
        if mid == 0:
            return grid_init(1, 1)
        out, _, _ = grid_init(mid, h)
        for y in range(h):
            for x in range(mid):
                out[y * mid + x] = color if (g[y * w + x] == color and g[y * w + mid + x] == color) else 0
        return out, mid, h
    return f

def make_copy_left(color):
    def f(g, w, h):
        mid = w // 2
        if mid == 0:
            return grid_init(1, 1)
        out, _, _ = grid_init(mid, h)
        for y in range(h):
            for x in range(mid):
                out[y * mid + x] = color if g[y * w + x] != 0 else 0
        return out, mid, h
    return f

def make_copy_right(color):
    def f(g, w, h):
        mid = w // 2
        if mid == 0:
            return grid_init(1, 1)
        out, _, _ = grid_init(mid, h)
        for y in range(h):
            for x in range(mid):
                out[y * mid + x] = color if g[y * w + mid + x] != 0 else 0
        return out, mid, h
    return f

def make_mask_intersection(color):
    def f(g, w, h):
        mid = w // 2
        if mid == 0:
            return grid_init(1, 1)
        out, _, _ = grid_init(mid, h)
        for y in range(h):
            for x in range(mid):
                out[y * mid + x] = color if (g[y * w + x] != 0 and g[y * w + mid + x] != 0) else 0
        return out, mid, h
    return f

def make_mask_union(color):
    def f(g, w, h):
        mid = w // 2
        if mid == 0:
            return grid_init(1, 1)
        out, _, _ = grid_init(mid, h)
        for y in range(h):
            for x in range(mid):
                out[y * mid + x] = color if (g[y * w + x] != 0 or g[y * w + mid + x] != 0) else 0
        return out, mid, h
    return f

def make_object_crop_largest(color_out=None):
    def f(g, w, h):
        comp = largest_component(g, w, h, 0)
        if comp is None:
            return grid_init(1, 1)
        xs = [p[0] for p in comp]
        ys = [p[1] for p in comp]
        mx, Mx = min(xs), max(xs)
        my, My = min(ys), max(ys)
        nw, nh = Mx - mx + 1, My - my + 1
        out, _, _ = grid_init(nw, nh)
        for y in range(nh):
            for x in range(nw):
                v = g[(my + y) * w + mx + x]
                if color_out is not None and v != 0:
                    v = color_out
                out[y * nw + x] = v
        return out, nw, nh
    return f

# ── candidate generation ───────────────────────────────────────────────────

def gen_cands(task):
    """Generate candidate transform functions for a task."""
    cands = []

    cands.append(('rotate90', op_rotate90))
    cands.append(('rotate180', op_rotate180))
    cands.append(('rotate270', op_rotate270))
    cands.append(('flip_horizontal', op_flip_horizontal))
    cands.append(('flip_vertical', op_flip_vertical))
    cands.append(('transpose', op_transpose))

    if not task.get('train'):
        return cands

    sample_w = len(task['train'][0]['input'][0]) if task['train'][0]['input'] else 0
    sample_h = len(task['train'][0]['input'])
    sample_ow = len(task['train'][0]['output'][0]) if task['train'][0]['output'] else 0

    # Scale 1x-4x
    for factor in range(1, 5):
        cands.append((f'scale_{factor}x', make_scale(factor)))

    # Crop all backgrounds
    for bg in range(10):
        cands.append((f'crop_background_{bg}', make_crop_background(bg)))

    # Crop + recolor all combos
    for bg in range(10):
        for c in range(1, 10):
            cands.append((f'crop_recolor_bg{bg}_c{c}', make_crop_background_recolor(bg, c)))

    # Foreground recolor all combos
    for bg in range(10):
        for c in range(1, 10):
            cands.append((f'foreground_recolor_bg{bg}_c{c}', make_foreground_recolor(bg, c)))

    # Translate all dx,dy from -5 to +5
    for dx in range(-5, 6):
        for dy in range(-5, 6):
            if dx == 0 and dy == 0:
                continue
            cands.append((f'translate_{dx}_{dy}', make_translate(dx, dy)))

    # Translate + recolor for observed colors
    observed_colors = set()
    for pair in task.get('train', []):
        for row in pair['input']:
            for v in row:
                if v != 0:
                    observed_colors.add(v)
        for row in pair['output']:
            for v in row:
                if v != 0:
                    observed_colors.add(v)
    obs = list(observed_colors)[:5]

    for ic in obs:
        for oc in obs:
            if ic == oc:
                continue
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    cands.append((f'translate_recolor_{dx}_{dy}_{ic}_{oc}',
                                  make_translate_recolor(dx, dy, ic, oc)))

    # Bbox outline & fill
    for bg in range(10):
        for c in range(1, 10):
            cands.append((f'bbox_outline_bg{bg}_c{c}', make_bbox_outline(bg, c)))
            cands.append((f'bbox_fill_bg{bg}_c{c}', make_bbox_fill(bg, c)))

    # Layer operations
    for c in range(1, 10):
        cands.append((f'layer_overlap_c{c}', make_layer_op('overlap', c)))
        cands.append((f'layer_union_c{c}', make_layer_op('union', c)))
        cands.append((f'layer_xor_c{c}', make_layer_op('xor', c)))
        cands.append((f'layer_left_only_c{c}', make_layer_op('left_only', c)))
        cands.append((f'layer_right_only_c{c}', make_layer_op('right_only', c)))
        cands.append((f'object_select_overlap_c{c}', make_object_select_overlap(c)))
        cands.append((f'copy_left_c{c}', make_copy_left(c)))
        cands.append((f'copy_right_c{c}', make_copy_right(c)))
        cands.append((f'mask_intersection_c{c}', make_mask_intersection(c)))
        cands.append((f'mask_union_c{c}', make_mask_union(c)))

    # Object crop largest
    cands.append(('object_crop_largest', make_object_crop_largest(None)))
    for c in range(1, 10):
        cands.append((f'object_crop_largest_recolor_c{c}', make_object_crop_largest(c)))
        cands.append((f'object_mask_largest_c{c}', make_object_crop_largest(c)))

    # Cross-pair color map
    cross_map = extract_cross_pair_color_map(task)
    if cross_map is not None:
        cands.append(('color_map_cross', make_color_map(cross_map)))

    # Per-pair color maps
    for pi, cmap in enumerate(extract_color_maps(task)):
        cands.append((f'color_map_pair{pi}', make_color_map(cmap)))

    # Color map subsets
    for pi, cmap in enumerate(extract_color_maps(task)):
        in_colors = [k for k in cmap if k != 0]
        if 1 <= len(in_colors) <= 5:
            for r in range(1, min(len(in_colors), 4) + 1):
                for subset in itertools.combinations(in_colors, r):
                    submap = {k: cmap[k] for k in subset}
                    cands.append((f'color_map_p{pi}_s{list(subset)}', make_color_map(submap)))

    # Panel selection: strip left columns
    if sample_w > sample_ow:
        for panel_w in range(1, sample_w - sample_ow + 1):
            cands.append((f'panel_select_left{panel_w}', make_panel_selection(panel_w)))

    # Bbox panel selection
    for pw in range(2, 6):
        for ph in range(2, 6):
            cands.append((f'bbox_panel_{pw}x{ph}', make_bbox_panel_selection(pw, ph)))

    return cands


def certify(task, fn):
    """Check if fn is consistent with ALL training pairs."""
    checked_cells = 0
    mismatches = 0

    for pair in task.get('train', []):
        inp = pair['input']
        out = pair['output']
        ih, iw = len(inp), len(inp[0]) if inp else 0
        oh, ow = len(out), len(out[0]) if out else 0
        inp_flat = [v for row in inp for v in row]
        out_flat = [v for row in out for v in row]

        try:
            pred, pw, ph = fn(inp_flat, iw, ih)
        except Exception:
            return False, 0, 0, 1

        if pw != ow or ph != oh:
            return False, checked_cells, 1, 0

        cells = pw * ph
        for i in range(cells):
            checked_cells += 1
            if pred[i] != out_flat[i]:
                mismatches += 1
                return False, checked_cells, mismatches, 0

    return True, checked_cells, mismatches, 0


def apply_to_test(task, fn):
    """Apply a certified function to test inputs."""
    results = []
    for pair in task.get('test', []):
        inp = pair['input']
        ih, iw = len(inp), len(inp[0]) if inp else 0
        inp_flat = [v for row in inp for v in row]
        try:
            pred, pw, ph = fn(inp_flat, iw, ih)
            results.append(_to_2d(pred, pw, ph))
        except Exception:
            results.append(None)
    return results
