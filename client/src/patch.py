# patch_onnx_graph.py
# Python 3.10+
# Требуется: pip install onnx onnxruntime onnxoptimizer (необязательно)

import onnx
from onnx import helper, numpy_helper, AttributeProto, TensorProto, GraphProto, ValueInfoProto
from typing import List

def patch_model(input_path: str, output_path: str) -> None:
    """
    Патчит ONNX-модель:
      - Заменяет узлы Ceil -> Floor
      - Убирает ceil_mode у AveragePool (ставит 0)
      - Корректирует Resize атрибуты, чтобы не триггерить ceil в shape computation
    """
    model = onnx.load(input_path)
    graph = model.graph

    # 1) Ceil -> Floor
    for node in graph.node:
        if node.op_type == "Ceil":
            node.op_type = "Floor"

    # 2) AveragePool ceil_mode -> 0
    for node in graph.node:
        if node.op_type in ("AveragePool", "MaxPool"):
            for attr in node.attribute:
                if attr.name == "ceil_mode" and attr.type == AttributeProto.INT:
                    if attr.i != 0:
                        attr.i = 0

    # 3) Resize: корректировка coordinate_transformation_mode и rounding_mode
    # - Меняем coordinate_transformation_mode на "half_pixel" если было "align_corners"
    # - Если есть rounding_mode, ставим "floor"
    for node in graph.node:
        if node.op_type == "Resize":
            has_ctm = False
            for attr in node.attribute:
                if attr.name == "coordinate_transformation_mode" and attr.type == AttributeProto.STRING:
                    has_ctm = True
                    val = attr.s.decode("utf-8").lower()
                    if val == "align_corners":
                        # WebGPU часто лучше переваривает half_pixel
                        attr.s = b"half_pixel"
                if attr.name == "rounding_mode" and attr.type == AttributeProto.STRING:
                    # Ставим floor, чтобы исключить ceil-ветки
                    attr.s = b"floor"
            # если coordinate_transformation_mode отсутствует, можно явно задать безопасный режим
            if not has_ctm:
                node.attribute.append(helper.make_attribute("coordinate_transformation_mode", "half_pixel"))

    # 4) Сохранить
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"Patched model saved to: {output_path}")

if __name__ == "__main__":
    # import argparse
    # p = argparse.ArgumentParser(description="Patch ONNX to avoid Ceil in shape computation for WebGPU.")
    # p.add_argument("--in", dest="inp", required=True, help="Input ONNX path")
    # p.add_argument("--out", dest="outp", required=True, help="Output ONNX path")
    # args = p.parse_args()
    in_p = "D://user//workspace//t1_tech_hack//video-stream-segmenetation//client//src//assets//fc_lmk.onnx"
    out_p = "D://user//workspace//t1_tech_hack//video-stream-segmenetation//client//src//assets//fc_patched.onnx"
    patch_model(in_p, out_p)
