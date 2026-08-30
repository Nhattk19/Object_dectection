"""Streamlit demo for the local Ultralytics YOLO model."""

from __future__ import annotations

import html
import sys
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError


WEB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEB_DIR.parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolo" / "best.pt"
SAMPLE_IMAGE = WEB_DIR / "assets" / "visdrone_sample.png"
DEFAULT_CONFIDENCE = 0.04
DEFAULT_MAX_DETECTIONS = 300
DEFAULT_NMS_IOU = 0.30

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from web.inference import (  # noqa: E402
    detection_color,
    load_model,
    model_metadata,
    predict,
    summarize_detections,
)


st.set_page_config(
    page_title="YOLO Vision Lab",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_styles() -> None:
    css = (WEB_DIR / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_model(model_path: str):
    return load_model(model_path)


@st.cache_data(show_spinner=False)
def get_model_metadata(model_path: str) -> dict:
    return model_metadata(model_path)


def read_uploaded_image(uploaded_file) -> Image.Image | None:
    if uploaded_file is None:
        return None
    try:
        image = Image.open(uploaded_file)
        return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("Không thể đọc ảnh này. Hãy thử file JPG, JPEG, PNG hoặc WEBP hợp lệ.")
        return None


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def render_topbar(model_ready: bool) -> None:
    status_text = "MODEL READY" if model_ready else "MODEL MISSING"
    status_class = "status-ready" if model_ready else "status-error"
    st.markdown(
        f"""
        <div class="topbar">
            <div class="wordmark">
                <span class="wordmark-box">DV</span>
                <span>YOLO VISION LAB</span>
            </div>
            <div class="model-status {status_class}">{status_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(metadata: dict) -> None:
    copy_column, visual_column = st.columns([1.08, 0.92], gap="large")
    with copy_column:
        st.markdown(
            """
            <section class="hero-copy">
                <p class="eyebrow">VISDRONE YOLO</p>
                <h1>Nhìn thành phố<br><span>qua từng vật thể.</span></h1>
                <p class="hero-subcopy">
                    Tải ảnh lên, chạy checkpoint YOLO local và kiểm tra từng dự đoán ngay trên trình duyệt.
                </p>
                <div class="hero-actions">
                    <a class="primary-link" href="#demo">Thử mô hình</a>
                    <span class="secondary-note">10 lớp giao thông</span>
                </div>
            </section>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="stat-rail">
                <div><strong>{metadata['classes']}</strong><span>Lớp vật thể</span></div>
                <div><strong>{metadata['image_size']}</strong><span>Input size</span></div>
                <div><strong>{metadata['weights_mb']:.0f} MB</strong><span>Local weights</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with visual_column:
        st.image(
            str(SAMPLE_IMAGE),
            caption="Ảnh mẫu đô thị dùng để thử checkpoint local.",
            width="stretch",
        )


def render_class_strip(labels: tuple[str, ...]) -> None:
    class_items = "".join(
        f'<span class="class-chip">{html.escape(label)}</span>' for label in labels
    )
    st.markdown(
        f"""
        <section class="class-section">
            <div>
                <h2>Mười lớp. Một khung nhìn.</h2>
                <p>Checkpoint được huấn luyện cho các vật thể giao thông phổ biến trong VisDrone.</p>
            </div>
            <div class="class-strip">{class_items}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_detection_summary(prediction) -> None:
    detections = prediction.detections
    unique_classes = len({item.label for item in detections})
    st.markdown(
        f"""
        <div class="result-rail">
            <div><strong>{len(detections)}</strong><span>Vật thể</span></div>
            <div><strong>{unique_classes}</strong><span>Lớp xuất hiện</span></div>
            <div><strong>{prediction.elapsed_ms:.0f} ms</strong><span>Inference</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    counts = summarize_detections(detections)
    if counts:
        label_ids = {item.label: item.label_id for item in detections}
        rows = "".join(
            f'<span class="count-chip">'
            f'<i class="legend-swatch" style="--legend-color: {detection_color(label_ids[label])}"></i>'
            f'<b>{html.escape(label)}</b><span>{count}</span></span>'
            for label, count in counts.items()
        )
        st.markdown(
            f'<p class="legend-label">Chú thích màu</p><div class="count-strip">{rows}</div>',
            unsafe_allow_html=True,
        )


def render_demo() -> None:
    st.markdown('<span id="demo"></span>', unsafe_allow_html=True)
    st.markdown(
        """
        <section class="section-heading">
            <h2>Chạy nhận diện trên ảnh của bạn</h2>
            <p>YOLO xử lý toàn bộ ảnh bằng checkpoint local. Điều chỉnh ba tham số trước khi chạy.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        source_tab, sample_tab = st.tabs(["Tải ảnh", "Dùng ảnh mẫu"])
        uploaded_image = None
        with source_tab:
            uploaded_file = st.file_uploader(
                "Chọn ảnh để nhận diện",
                type=["jpg", "jpeg", "png", "webp"],
                help="Ảnh có nhiều phương tiện hoặc người đi bộ sẽ cho kết quả trực quan hơn.",
            )
            uploaded_image = read_uploaded_image(uploaded_file)
        with sample_tab:
            st.markdown(
                "Ảnh mẫu là cảnh giao thông nhìn từ trên cao, phù hợp để kiểm tra nhanh cả xe và người."
            )
            use_sample = st.checkbox("Chọn ảnh mẫu", value=uploaded_image is None)

        selected_image = uploaded_image
        source_name = "uploaded-image"
        if selected_image is None and use_sample:
            selected_image = Image.open(SAMPLE_IMAGE).convert("RGB")
            source_name = "visdrone-sample"

        confidence_column, max_det_column, nms_column = st.columns(3, gap="medium")
        with confidence_column:
            confidence_threshold = st.slider(
                "Ngưỡng confidence",
                min_value=0.0,
                max_value=1.0,
                value=DEFAULT_CONFIDENCE,
                step=0.01,
                format="%.2f",
                help="Chỉ giữ bbox có confidence lớn hơn hoặc bằng ngưỡng này.",
            )
        with max_det_column:
            max_detections = st.slider(
                "Số bbox tối đa",
                min_value=1,
                max_value=1000,
                value=DEFAULT_MAX_DETECTIONS,
                step=1,
                help="Giới hạn số bbox YOLO trả về sau NMS.",
            )
        with nms_column:
            nms_iou_threshold = st.slider(
                "NMS IoU",
                min_value=0.0,
                max_value=1.0,
                value=DEFAULT_NMS_IOU,
                step=0.05,
                format="%.2f",
                help="IoU dùng bởi NMS tích hợp của YOLO để loại bbox trùng.",
            )

        st.caption(
            f"Đang chọn: confidence {confidence_threshold:.2f} · tối đa "
            f"{max_detections} bbox · NMS IoU {nms_iou_threshold:.2f}."
        )
        run_inference = st.button(
            "Chạy nhận diện",
            type="primary",
            width="stretch",
            disabled=selected_image is None,
        )

    if selected_image is None:
        st.info("Tải một ảnh hoặc chọn ảnh mẫu để bắt đầu.")
        return
    if not run_inference:
        st.markdown('<p class="preview-label">Ảnh đang chọn</p>', unsafe_allow_html=True)
        st.image(selected_image, width="stretch")
        return

    try:
        with st.spinner("Đang chạy YOLO trên toàn bộ ảnh..."):
            prediction = predict(
                selected_image,
                get_model(str(MODEL_PATH)),
                threshold=confidence_threshold,
                max_detections=max_detections,
                nms_iou_threshold=nms_iou_threshold,
            )
    except Exception as error:
        st.error(f"Không thể chạy mô hình: {error}")
        return

    render_detection_summary(prediction)
    st.caption(
        f"Toàn bộ ảnh (1 lần suy luận) · confidence {confidence_threshold:.2f} · "
        f"NMS IoU {nms_iou_threshold:.2f} · {len(prediction.detections)} bbox."
    )
    original_column, result_column = st.columns(2, gap="medium")
    with original_column:
        st.markdown("#### Ảnh gốc")
        st.image(selected_image, width="stretch")
    with result_column:
        st.markdown("#### Kết quả · Toàn bộ ảnh")
        st.image(prediction.image, width="stretch")

    st.download_button(
        "Tải ảnh kết quả",
        data=image_to_png_bytes(prediction.image),
        file_name=f"{source_name}-yolo-full.png",
        mime="image/png",
        width="stretch",
    )
    if not prediction.detections:
        st.warning("Không có vật thể vượt qua ngưỡng hiện tại. Hãy giảm confidence và thử lại.")


def render_footer(metadata: dict) -> None:
    st.markdown(
        f"""
        <footer class="site-footer">
            <div>
                <strong>YOLO Vision Lab</strong>
                <span>Streamlit demo dùng checkpoint local.</span>
            </div>
            <div>{html.escape(metadata['architecture'])} / {html.escape(metadata['backbone'])}</div>
        </footer>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_styles()
    model_ready = MODEL_PATH.is_file()
    try:
        metadata = get_model_metadata(str(MODEL_PATH))
    except (FileNotFoundError, OSError, ValueError) as error:
        render_topbar(False)
        st.error(f"Không thể đọc checkpoint {MODEL_PATH}: {error}")
        st.stop()

    render_topbar(model_ready)
    render_hero(metadata)
    render_class_strip(metadata["labels"])
    render_demo()
    render_footer(metadata)


if __name__ == "__main__":
    main()
