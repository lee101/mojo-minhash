from std.collections import List

comptime U64Ptr = Pointer[UInt64, AnyOrigin[mut=True]]


def probe(p: U64Ptr) -> UInt64:
    var v = UInt64(3)
    var old = p.atomic_cas(0, v, 0)
    var old2 = p.atomic_compare_exchange(0, v, 0)
    var old3 = p.compare_and_swap(0, v, 0)
    var old4 = p.atomic_min(0, v)
    return old + old2 + old3 + UInt64(old4)


def main():
    print("probe")
