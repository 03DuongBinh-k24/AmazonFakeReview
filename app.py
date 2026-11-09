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
        st.write("Tải CSV hoặc XLSX có các cột `text`, `rating`, `category`. Dòng dữ liệu lỗi được giữ trong báo cáo và không đưa vào suy luận.")
        st.info("Tối đa 5.000 dòng dữ liệu mỗi lượt để kiểm soát khối lượng xử lý. Nếu vượt giới hạn, hãy chia tệp thành nhiều phần rồi tải lại.")
        uploaded = st.file_uploader("Tệp đánh giá", type=["csv", "xlsx"])
        file_key = (uploaded.name, hashlib.sha256(uploaded.getbuffer()).digest()) if uploaded is not None else None
        if st.session_state.get("batch_file_key") != file_key:
            st.session_state.pop("batch_output", None)
            st.session_state["batch_file_key"] = file_key

        if uploaded is not None and st.button("Phân tích tệp", type="primary"):
            st.session_state.pop("batch_output", None)
            try:
                st.session_state["batch_output"] = run_batch(uploaded, bundle, stop_words)
            except Exception as exc:
                st.error(f"Không thể xử lý tệp: {exc}")

        batch = st.session_state.get("batch_output")
        if batch is not None:
            valid = batch[batch["prediction_label"] != ""]
            counts = valid["action"].value_counts().reindex(ROUTING_ACTIONS, fill_value=0)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Tổng dòng", f"{len(batch):,}")
            c2.metric("Đã xử lý", f"{len(valid):,}")
            c3.metric("Chuyển hậu kiểm", f"{int(counts[REVIEW_ACTION]):,}")
            c4.metric("Dòng lỗi", f"{len(batch) - len(valid):,}")
            elapsed = batch.attrs["inference_seconds"]
            st.caption(f"Thời gian tiền xử lý và suy luận: {elapsed:.3f} giây")

            chart_data = pd.DataFrame({"Nhóm": counts.index, "Số đánh giá": counts.values})
            st.subheader("Phân bố theo khuyến nghị xử lý")
            st.vega_lite_chart(
                chart_data,
                {
                    "mark": {"type": "arc", "innerRadius": 45},
                    "encoding": {
                        "theta": {"field": "Số đánh giá", "type": "quantitative"},
                        "color": {
                            "field": "Nhóm",
                            "type": "nominal",
                            "scale": {"domain": ROUTING_ACTIONS, "range": ["#2e8b57", "#c0392b", "#e6a700"]},
                        },
                        "tooltip": [{"field": "Nhóm", "type": "nominal"}, {"field": "Số đánh giá", "type": "quantitative"}],
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

    with about_tab:
        st.subheader("Phạm vi áp dụng")
        st.write(
            "Ứng dụng hỗ trợ chuyên viên sàng lọc đánh giá sản phẩm Amazon bằng tiếng Anh. "
            "Mô hình nhận nội dung, điểm từ 1 đến 5 sao và danh mục sản phẩm. "
            f"Bộ mã hóa có {len(known_categories)} danh mục từng xuất hiện trong tập huấn luyện."
        )
        with st.expander("Danh mục trong tập huấn luyện"):
            st.write(", ".join(known_categories))
        st.write(
            "Ứng dụng vẫn trả dự đoán kèm cảnh báo nếu gặp danh mục mới. "
            "Hiệu năng trên các ngôn ngữ, nền tảng hoặc danh mục ngoài phạm vi nghiên cứu cần được đánh giá riêng."
        )

        st.subheader("Mô hình và giới hạn")
        st.write(
            "LightGBM được chọn trong Chương 4. Quy trình trích xuất 14 đặc trưng ngữ phong ban đầu "
            "và giữ lại 13 sau bước chọn đặc trưng. Bộ tiền xử lý kết hợp các đặc trưng này với nội dung từ vựng, điểm sao và danh mục. "
            "Đầu vào sau biến đổi có 5.024 đặc trưng. Các trường đầu vào này không xác minh danh tính "
            "người viết hay giao dịch mua hàng."
        )

        st.subheader("Quy tắc phân luồng")
        st.table(pd.DataFrame([
            {"Điều kiện": "Dưới 20 từ hoặc 0,40 ≤ P(CG) ≤ 0,60", "Khuyến nghị": REVIEW_ACTION},
            {"Điều kiện": "Từ 20 từ trở lên và P(CG) < 0,40", "Khuyến nghị": ACCEPT_ACTION},
            {"Điều kiện": "Từ 20 từ trở lên và P(CG) > 0,60", "Khuyến nghị": CG_ACTION},
        ]))
        st.write("Ví dụ, đánh giá có 16 từ và P(CG) = 0,97 vẫn được đề xuất hậu kiểm do văn bản ngắn.")

        st.subheader("Diễn giải kết quả")
        st.write(
            "P(CG) là xác suất mô hình ước tính cho lớp máy sinh. TreeSHAP cho biết đặc trưng nào "
            "đẩy dự đoán về phía CG hoặc OR. Chuyên viên kết hợp các tín hiệu này với ngữ cảnh "
            "để quyết định cách xử lý đánh giá."
        )


if __name__ == "__main__":
    main()
