import math

import numpy as np

VIEW_LENGTH = 400  # μm
VIEW_PIXEL = 384
CNBD_MIN_LENGTH = 20


def get_view_area_mm2():
    return (VIEW_LENGTH / 1000) ** 2


def pixel_to_mm(num):
    return num * (VIEW_LENGTH / 1000) / VIEW_PIXEL


def get_length(p):
    return np.sum(s.length for s in p.segments)


def get_nerve_num(p):
    return len(p.nerves)


def get_nerve_branch_num(p):
    branch_records, branch_count = [], 0
    for nerve in p.nerves:
        for node in [n for n in nerve.nodes if n.class_node == 'branch']:
            branches = [s.index for s in node.neighbors if p.segments[s.index].class_segment == 'branch']
            branches = [si for si in branches if si not in branch_records]
            branches = [si for si in branches if p.segments[si].length > CNBD_MIN_LENGTH]
            branch_count += len(branches)
            branch_records.extend(branches)

    return branch_count

def get_CNFL(p):
    return pixel_to_mm(get_length(p)) / get_view_area_mm2()


def get_CNFD(p):
    return get_nerve_num(p) / get_view_area_mm2()


def get_CNBD(p):
    return get_nerve_branch_num(p) / get_view_area_mm2()


def get_summary(p):
    summary_text = f"""Summary:
    CNF length of pixel: {get_length(p)}
    CNF length(mm2): {pixel_to_mm(get_length(p))}
    CNF trunk num: {get_nerve_num(p)}
    CNF primary branch num: {get_nerve_branch_num(p)}
    
    CNFL = {get_CNFL(p)}
    CNFD = {get_CNFD(p)}
    CNBD = {get_CNBD(p)}
    """
    print(summary_text)
