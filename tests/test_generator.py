from env.generator import assert_layout_consistent, gen_layout, instance_spec, load_split
from env.serialize import layout_from_json, layout_to_json, scene_text


def test_same_seed_is_byte_identical():
    a = gen_layout(7, 3, 2, 1, topology="l_shape", split="train")
    b = gen_layout(7, 3, 2, 1, topology="l_shape", split="train")
    assert layout_to_json(a) == layout_to_json(b)


def test_different_seeds_differ():
    a = gen_layout(1, 2, 1, 0, topology="chain")
    b = gen_layout(2, 2, 1, 0, topology="chain")
    assert layout_to_json(a) != layout_to_json(b)


def test_json_roundtrip():
    layout = gen_layout(3, 2, 2, 1, topology="chain", split="train")
    assert layout_from_json(layout_to_json(layout)) == layout


def test_split_distributions_and_invariants():
    for split, n in (("train", 6), ("test_id", 4), ("test_ood", 4)):
        layouts = load_split(split, n)
        assert len(layouts) == n
        for layout in layouts:
            assert_layout_consistent(layout)
            text = scene_text(layout)
            assert layout.rooms[0].name in text
            assert "Goal:" in text
            if split in ("train", "test_id"):
                assert layout.n_rooms in (2, 3)
                assert 1 <= layout.n_boxes <= 3
                assert layout.topology in ("chain", "l_shape")
            else:
                assert layout.n_rooms in (4, 5)
                assert 4 <= layout.n_boxes <= 6
                assert layout.topology in ("grid", "star")


def test_instance_spec_is_deterministic():
    assert instance_spec("test_id", 3) == instance_spec("test_id", 3)
    assert instance_spec("train", 0)[0] != instance_spec("test_ood", 0)[0]
