from scripts.canonicalize_genomes import merge_segment_list

def test_merge_segment_list():
    segments_list = []
    merged = merge_segment_list(segments_list)
    assert merged == []

    segments_list = [(0, 10)]
    merged = merge_segment_list(segments_list)
    assert merged == [(0, 10)]

    segments_list = [(0, 10), (10, 20)]
    merged = merge_segment_list(segments_list)
    assert merged == [(0, 20)]

    segments_list = [(0, 20), (30, 40)]
    merged = merge_segment_list(segments_list)
    assert merged == [(0, 20), (30, 40)]

    segments_list = [(0, 20), (25, 40)]
    merged = merge_segment_list(segments_list)
    assert merged == [(0, 20), (25, 40)]

    # I know the segments are already sorted with bisect.insort, so there's no need to iteratively merge multiple segments.
    segments_list = [(0, 20), (10, 50), (25, 40)]
    merged = merge_segment_list(segments_list)
    assert merged == [(0, 50)]