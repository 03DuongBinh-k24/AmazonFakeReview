from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from inference import APP_DIR, ensure_nltk_resources, infer_reviews, load_bundle, verify_preprocessing_reference


PRESET_PATH = APP_DIR / "preset_reviews.json"
UNCERTAINTY_LOW = 0.40
UNCERTAINTY_HIGH = 0.60
SHORT_REVIEW_WORDS = 20
MAX_BATCH_ROWS = 5000

st.set_page_config(page_title="Kiểm định đánh giá Amazon", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner="Đang nạp model và tài nguyên xử lý ngôn ngữ...")
def runtime():
    stop_words = ensure_nltk_resources()
    bundle = load_bundle()
    verify_preprocessing_reference(stop_words)
    return bundle, stop_words


def classification_status(prediction: int, probability: float) -> str:
    if UNCERTAINTY_LOW <= probability <= UNCERTAINTY_HIGH:
        return "Cần hậu kiểm"
    return "Máy sinh (CG)" if prediction == 1 else "Hợp lệ (OR)"


def operational_action(prediction: int, probability: float, word_count: int) -> str:
    if UNCERTAINTY_LOW <= probability <= UNCERTAINTY_HIGH or word_count < SHORT_REVIEW_WORDS:
        return "Chuyển hậu kiểm"
    return "Gỡ bỏ" if prediction == 1 else "Chấp thuận"


def category_options(bundle: dict) -> list[str]:
    encoder = bundle["preprocessor"].named_transformers_["category"]
    return [str(value) for value in encoder.categories_[0]]


def show_prediction(review: pd.DataFrame, bundle: dict, stop_words: set[str], show_xai: bool = True):
    started = time.perf_counter()
    raw14, pred, prob, contributions = infer_reviews(review, bundle, stop_words, explain=show_xai)
    elapsed_ms = (time.perf_counter() - started) * 1000

    word_count = int(raw14.iloc[0]["word_count"])
    probability = float(prob[0])
    prediction = int(pred[0])
    label = "CG – Máy sinh" if prediction == 1 else "OR – Người viết"
    status = classification_status(prediction, probability)
    action = operational_action(prediction, probability, word_count)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Nhãn mô hình", label)
    col_b.metric("Xác suất CG", f"{probability:.1%}")
    col_c.metric("Trạng thái vận hành", status)
    st.progress(min(max(probability, 0.0), 1.0), text="Xác suất đánh giá thuộc lớp CG")
    st.caption(f"Khuyến nghị: **{action}** · Thời gian xử lý đầu-cuối: {elapsed_ms:.1f} ms")

    if word_count < SHORT_REVIEW_WORDS:
        st.warning(
            f"Đánh giá có {word_count} từ, dưới ngưỡng {SHORT_REVIEW_WORDS}. "
            "Các đặc trưng ngôn ngữ có thể kém ổn định; cần xem xét thêm thông tin xác minh."
        )
    if UNCERTAINTY_LOW <= probability <= UNCERTAINTY_HIGH:
        st.warning(
            f"Xác suất CG nằm trong vùng bất định [{UNCERTAINTY_LOW:.2f}, {UNCERTAINTY_HIGH:.2f}]; "
            "mẫu được chuyển sang hậu kiểm thay vì kết luận tự động."
        )
    if str(review.iloc[0]["category"]) not in category_options(bundle):
        st.info("Danh mục chưa xuất hiện trong tập huấn luyện; khối mã hóa danh mục được để toàn số 0 theo cấu hình model.")

    with st.expander("Chỉ số ngữ phong của đánh giá"):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Số từ", f"{word_count}")
        m2.metric("TTR", f"{raw14.iloc[0]['ttr']:.3f}")
        m3.metric("RTTR", f"{raw14.iloc[0]['rttr']:.3f}")
        m4.metric("Tỷ lệ từ dừng", f"{raw14.iloc[0]['stopword_ratio']:.3f}")

    if show_xai and contributions is not None:
        with st.expander("Đóng góp đặc trưng cục bộ (TreeSHAP)", expanded=True):
            shap_values = contributions[0]
            names = bundle["feature_names"]
            top_indices = np.argsort(np.abs(shap_values))[-12:][::-1]
            explanation = pd.DataFrame(
                {
                    "Đặc trưng": [str(names[index]) for index in top_indices],
                    "Đóng góp": [float(shap_values[index]) for index in top_indices],
                }
            )
            st.bar_chart(explanation.set_index("Đặc trưng")["Đóng góp"])
            st.dataframe(explanation, hide_index=True, width="stretch")
            st.caption("Giá trị dương đẩy dự đoán về CG; giá trị âm đẩy dự đoán về OR. Đây là giải thích đóng góp của mô hình, không phải quan hệ nhân quả.")


def normalize_batch_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    aliases = {"text_": "text", "review": "text", "review_text": "text", "product_category": "category", "stars": "rating"}
    frame = frame.rename(columns={key: value for key, value in aliases.items() if key in frame.columns and value not in frame.columns})
    missing = {"text", "rating", "category"} - set(frame.columns)
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {', '.join(sorted(missing))}")
    return frame


def run_batch(uploaded, bundle: dict, stop_words: set[str]):
    if uploaded.name.lower().endswith(".csv"):
        source = pd.read_csv(uploaded)
    else:
        source = pd.read_excel(uploaded, engine="openpyxl")
    source = normalize_batch_columns(source)
    if len(source) > MAX_BATCH_ROWS:
        raise ValueError(f"Tệp có {len(source):,} dòng; giới hạn mỗi lượt là {MAX_BATCH_ROWS:,} dòng.")

    result = source.copy()
    result.insert(0, "source_row", np.arange(2, len(result) + 2))
    result["prediction_label"] = ""
    result["probability_cg"] = np.nan
    result["status"] = "Lỗi dữ liệu đầu vào"
    result["word_count"] = np.nan
    result["action"] = "Kiểm tra dữ liệu"
    result["warning"] = ""

    valid_positions = []
    valid_rows = []
    for idx, row in source.iterrows():
        text = "" if pd.isna(row["text"]) else str(row["text"])
        category = "" if pd.isna(row["category"]) else str(row["category"]).strip()
        try:
            rating = float(row["rating"])
        except (TypeError, ValueError):
            rating = np.nan
        errors = []
        if not text.strip():
            errors.append("Thiếu nội dung")
        if not category:
            errors.append("Thiếu category")
        if not np.isfinite(rating) or rating < 1 or rating > 5:
            errors.append("Rating phải là số từ 1 đến 5")
        if errors:
            result.loc[idx, "warning"] = "; ".join(errors)
            continue
        valid_positions.append(idx)
        valid_rows.append({"text": text, "rating": rating, "category": category})

    if valid_rows:
        started = time.perf_counter()
        raw14, predictions, probabilities, _ = infer_reviews(pd.DataFrame(valid_rows), bundle, stop_words, explain=False)
        elapsed = time.perf_counter() - started
        for offset, source_idx in enumerate(valid_positions):
            p = float(probabilities[offset])
            prediction = int(predictions[offset])
            count = int(raw14.iloc[offset]["word_count"])
            is_uncertain = UNCERTAINTY_LOW <= p <= UNCERTAINTY_HIGH
            is_short = count < SHORT_REVIEW_WORDS
            label = "CG" if prediction == 1 else "OR"
            result.loc[source_idx, "prediction_label"] = label
            result.loc[source_idx, "probability_cg"] = p
            result.loc[source_idx, "status"] = classification_status(prediction, p)
            result.loc[source_idx, "word_count"] = count
            result.loc[source_idx, "action"] = operational_action(prediction, p, count)
            warnings = []
            if is_short:
                warnings.append(f"Văn bản dưới {SHORT_REVIEW_WORDS} từ")
            if is_uncertain:
                warnings.append("Xác suất trong vùng bất định")
            if str(valid_rows[offset]["category"]) not in category_options(bundle):
                warnings.append("Category chưa thấy trong train")
            result.loc[source_idx, "warning"] = "; ".join(warnings)
    else:
        elapsed = 0.0

    result.attrs["inference_seconds"] = elapsed
    return result


def export_xlsx(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name="Ket_qua")
        summary = (
            frame[frame["prediction_label"] != ""]
            .groupby(["status", "action"], dropna=False)
            .size()
            .reset_index(name="so_luong")
        )
        summary.to_excel(writer, index=False, sheet_name="Tong_hop")
    return buffer.getvalue()


def main():
    st.title("Hệ thống hỗ trợ sàng lọc đánh giá giả mạo")
    st.caption("Kết quả dựa trên nội dung, rating và danh mục. Đây là công cụ hỗ trợ rà soát; các trường hợp cảnh báo cần được chuyên viên xem xét.")

    try:
        bundle, stop_words = runtime()
    except Exception as exc:
        st.error(f"Không thể khởi tạo mô hình hoặc tài nguyên tiền xử lý: {exc}")
        st.stop()

    single_tab, batch_tab, about_tab = st.tabs(["Kiểm định đơn lẻ", "Kiểm định hàng loạt", "Thông tin mô hình"])

    with single_tab:
        presets = json.loads(PRESET_PATH.read_text(encoding="utf-8"))
        preset_names = ["Không chọn preset"] + [item["name"] for item in presets]
        selected = st.selectbox("Kịch bản kiểm chứng có nguồn", preset_names)
        if st.button("Nạp nội dung mẫu", disabled=selected == "Không chọn preset"):
            item = next(p for p in presets if p["name"] == selected)
            st.session_state["review_text"] = item["text"]
            st.session_state["review_rating"] = int(item["rating"])
            st.session_state["review_category"] = item["category"]
            st.session_state["review_category_choice"] = item["category"]
            st.session_state["preset_source"] = item["source_review_id"]

        known_categories = category_options(bundle)
        default_category = st.session_state.get("review_category", known_categories[0])
        if default_category not in known_categories:
            default_category = known_categories[0]

        text = st.text_area("Nội dung đánh giá", key="review_text", height=180, placeholder="Nhập nội dung cần kiểm tra")
        rating = st.select_slider("Điểm đánh giá", options=[1, 2, 3, 4, 5], value=int(st.session_state.get("review_rating", 5)), key="review_rating")
        category_choice = st.selectbox("Danh mục", known_categories + ["Khác / chưa biết"], index=known_categories.index(default_category), key="review_category_choice")
        if category_choice == "Khác / chưa biết":
            category = st.text_input("Nhập mã category", value="__unknown_category__") or "__unknown_category__"
        else:
            category = category_choice

        if st.button("Phân tích đánh giá", type="primary"):
            if not text.strip():
                st.error("Hãy nhập nội dung trước khi phân tích.")
            else:
                review = pd.DataFrame([{"text": text, "rating": rating, "category": category}])
                try:
                    show_prediction(review, bundle, stop_words, show_xai=True)
                    if st.session_state.get("preset_source"):
                        st.caption(f"Preset nguồn review_id={st.session_state['preset_source']} (mẫu thực từ tập kiểm tra; chỉ dùng minh họa chức năng).")
                except Exception as exc:
                    st.error(f"Không thể xử lý đánh giá: {exc}")

    with batch_tab:
        st.write("Tải CSV hoặc XLSX có các cột `text`, `rating`, `category`. Dòng dữ liệu lỗi được giữ trong báo cáo và không đưa vào suy luận. Khuyến nghị trong tệp là gợi ý rà soát, không tự động chấp thuận hoặc gỡ đánh giá.")
        uploaded = st.file_uploader("Tệp đánh giá", type=["csv", "xlsx"])
        if uploaded is not None and st.button("Phân tích tệp", type="primary"):
            try:
                batch = run_batch(uploaded, bundle, stop_words)
                valid = batch[batch["prediction_label"] != ""]
                st.session_state["batch_output"] = batch
                st.session_state["batch_elapsed"] = batch.attrs.get("inference_seconds", 0.0)
                total = len(batch)
                counts = valid["status"].value_counts()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Tổng dòng", f"{total:,}")
                c2.metric("Đã xử lý", f"{len(valid):,}")
                c3.metric("Cần hậu kiểm", f"{int(counts.get('Cần hậu kiểm', 0)):,}")
                c4.metric("Dòng lỗi", f"{total-len(valid):,}")
                st.caption(f"Thời gian tiền xử lý và suy luận: {st.session_state['batch_elapsed']:.3f} giây")

                risk = valid["status"].map(
                    lambda value: "Cần hậu kiểm" if value == "Cần hậu kiểm" else "Dự đoán CG" if value == "Máy sinh (CG)" else "Dự đoán OR"
                ).value_counts().reindex(["Dự đoán OR", "Dự đoán CG", "Cần hậu kiểm"], fill_value=0)
                chart_data = pd.DataFrame({"Nhóm": risk.index, "Số đánh giá": risk.values})
                st.subheader("Cơ cấu kết quả sàng lọc")
                st.vega_lite_chart(
                    chart_data,
                    {
                        "mark": {"type": "arc", "innerRadius": 45},
                        "encoding": {
                            "theta": {"field": "Số đánh giá", "type": "quantitative"},
                            "color": {"field": "Nhóm", "type": "nominal", "scale": {"domain": ["Dự đoán OR", "Dự đoán CG", "Cần hậu kiểm"], "range": ["#2e8b57", "#c0392b", "#e6a700"]}},
                            "tooltip": [{"field": "Nhóm"}, {"field": "Số đánh giá"}],
                        },
                    },
                    use_container_width=True,
                )
                st.dataframe(batch, hide_index=True, width="stretch")
                st.download_button(
                    "Tải báo cáo kết quả XLSX",
                    data=export_xlsx(batch),
                    file_name="ket_qua_tham_dinh.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except Exception as exc:
                st.error(f"Không thể xử lý tệp: {exc}")

        if "batch_output" in st.session_state and uploaded is not None:
            st.download_button(
                "Tải lại báo cáo XLSX",
                data=export_xlsx(st.session_state["batch_output"]),
                file_name="ket_qua_tham_dinh.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_again",
            )

    with about_tab:
        st.write("Mô hình sàng lọc là LightGBM được chọn trong Chương 4. Mô hình nhận nội dung đánh giá, điểm sao và danh mục; biểu diễn đầu vào có 5.024 đặc trưng.")
        st.write("Ngưỡng vận hành: cảnh báo văn bản dưới 20 từ; chuyển hậu kiểm khi xác suất CG nằm trong khoảng 0,40–0,60.")
        st.info("Các ngưỡng là quy tắc hỗ trợ rà soát được thiết lập từ kết quả tập kiểm tra, không phải kết luận chắc chắn về nguồn gốc bài đánh giá.")


if __name__ == "__main__":
    main()
