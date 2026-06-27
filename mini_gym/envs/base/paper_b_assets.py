"""Runtime asset helpers for Paper B sphere-foot Mini Cheetah variants."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree


def foot_radius_buckets(radius_range, num_buckets=5):
    low, high = radius_range
    if num_buckets <= 1:
        return [0.5 * (low + high)]
    step = (high - low) / (num_buckets - 1)
    return [low + step * bucket_id for bucket_id in range(num_buckets)]


def generate_sphere_foot_urdf_variants(template_path, output_dir, radii):
    template_path = Path(template_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variant_paths = []
    for radius in radii:
        output_path = output_dir / f"{template_path.stem}_foot_radius_{int(round(radius * 1000)):03d}mm.urdf"
        _write_sphere_foot_variant(template_path, output_path, radius)
        variant_paths.append(output_path)
    return variant_paths


def _write_sphere_foot_variant(template_path, output_path, radius):
    tree = ElementTree.parse(template_path)
    root = tree.getroot()
    radius_text = f"{radius:.6f}".rstrip("0").rstrip(".")

    for link in root.findall("link"):
        if not link.attrib.get("name", "").endswith("_foot"):
            continue
        for sphere in link.findall(".//sphere"):
            sphere.set("radius", radius_text)

    tree.write(output_path, encoding="utf-8", xml_declaration=True)
