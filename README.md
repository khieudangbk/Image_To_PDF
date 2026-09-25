# Image To PDF VN

Ứng dụng desktop chuyển **ảnh chụp / ảnh scan tài liệu** thành **file PDF văn bản thật**, giống như khi
xuất từ Word sang PDF. Trong file PDF, chữ là chữ thật (bôi đen, sao chép, tìm kiếm được) với phông chữ
nhúng, chứ không phải ảnh dán vào trang.

## Tính năng

- **OCR tiếng Việt** bằng Tesseract 5 với model `tessdata_best` (có thể chọn Việt + Anh).
- **Tiền xử lý ảnh tự động**: cắt mép và nắn phẳng ảnh chụp điện thoại, khử bóng/ánh sáng không đều,
  chỉnh nghiêng, tự xoay trang ngược/ngang, phóng ảnh độ phân giải thấp.
- **Dựng lại bố cục**: tiêu ngữ hai cột, tiêu đề, đoạn văn căn đều, danh sách, trang nhiều cột.
  Mỗi dòng được đặt đúng vị trí và cỡ chữ như bản gốc.
- **Bảng có đường kẻ**: phát hiện lưới, nhận dạng từng ô riêng, vẽ lại đường kẻ dạng vector.
- **Chữ đậm** (dựa trên độ dày nét chữ) và **tiêu đề** (thành mục lục/bookmark trong PDF).
- **Giữ hình**: logo, con dấu đỏ, chữ ký, ảnh chân dung được giữ dạng ảnh (nén JPEG), còn lại
  đều là văn bản. Chữ in màu (vd. tiêu đề đỏ) được giữ đúng màu; chữ đen bị con dấu đè lên vẫn
  đọc được.
- **Giữ nền và màu gốc** (tuỳ chọn, tab Cài đặt hoặc `--keep-background`): dùng ảnh gốc đã xoá
  chữ in làm nền — giữ nguyên hoa văn viền, nền chìm, màu giấy, con dấu — rồi vẽ chữ thật đè lên.
  Chữ nào OCR không chắc chắn thì giữ nguyên nét gốc và chỉ đặt lớp chữ ẩn để vẫn tìm kiếm được.
  Hợp với chứng chỉ, bằng cấp, giấy tờ có nền trang trí; file nặng hơn (~0,5 MB/trang).
- **Sửa dấu tiếng Việt theo ngữ cảnh** bằng từ điển (ví dụ `GIÁM ĐÓC → GIÁM ĐỐC`,
  `phô thông → phổ thông`, `Chuyên đôi số → Chuyển đổi số`) nhưng không đụng tới chữ đúng
  (`cấp tính`, `Nguyễn`, `hòa/hoà`…).
- **Chỉnh sửa trước khi xuất**: bấm vào một vùng trên ảnh hoặc bản xem trước để sửa chữ, cỡ chữ,
  in đậm, căn lề, loại khối; xoá vùng thừa.
- Nhiều trang: kéo thả ảnh/thư mục, sắp xếp thứ tự, xử lý song song nhiều trang.
- Xuất **PDF** (A4, Letter hoặc theo khổ ảnh; Times New Roman, Arial, Tahoma, Cambria, Courier New)
  và **TXT**.

Kết quả đo trên bộ ảnh mẫu (`tests/`): 98–99,6% số từ nhận đúng với ảnh scan, ảnh chụp nghiêng có
bóng và trang hai cột.

## Sử dụng bản đóng gói (không cần cài Python)

1. Giải nén `ImageToPDF-win64.zip`.
2. Chạy `ImageToPDF\ImageToPDF.exe`.
3. Kéo thả ảnh vào cửa sổ → bấm **Nhận dạng** (F5) → kiểm tra/sửa → **Xuất PDF** (Ctrl+E).

Xử lý hàng loạt bằng dòng lệnh:

```
ImageToPDF.exe --cli thu_muc_anh -o ket_qua.pdf
```

## Chạy từ mã nguồn

Yêu cầu: Windows 10/11, Python 3.11+, [Tesseract 5](https://github.com/UB-Mannheim/tesseract/wiki)
(model tiếng Việt đã có sẵn trong `models/tessdata`, không cần cài thêm gói ngôn ngữ).

```
pip install -r requirements.txt
python app.py                                  # giao diện
python cli.py anh1.jpg anh2.png -o ra.pdf      # dòng lệnh
python cli.py thu_muc/ -o ra.pdf --font Arial --paper Letter --lang vie+eng
```

Tuỳ chọn dòng lệnh: `--no-crop` (không tự cắt/nắn), `--no-tables`, `--no-figures`,
`--keep-background` (giữ nền và màu gốc).

## Đóng gói

```
powershell -ExecutionPolicy Bypass -File packaging\build.ps1
```

Tạo `dist\ImageToPDF\ImageToPDF.exe` và `dist\ImageToPDF-win64.zip` (kèm Tesseract rút gọn,
model tiếng Việt và từ điển).

## Kiểm thử

```
python tests\make_samples.py      # tạo ảnh mẫu có đáp án (scan, ảnh chụp, hai cột)
python tests\evaluate.py          # đo độ chính xác từng mẫu
python tests\test_vn_correct.py   # bộ sửa dấu: các ca phải sửa và không được sửa
python tests\test_orientation.py  # tự xoay 90/180/270°
python tests\test_gui.py          # giao diện đầu-cuối (offscreen)
python tests\debug_page.py anh.jpg  # xuất ảnh khung vùng + bản xem trước để soát lỗi
```

## Cấu trúc

| Thư mục/tệp | Vai trò |
|---|---|
| `img2pdf/preprocess.py` | Nắn phối cảnh, khử nền, chỉnh nghiêng, xoay, phóng ảnh |
| `img2pdf/ocr.py` | Gọi Tesseract, đọc hOCR (vị trí từng từ, đường chân chữ) |
| `img2pdf/tables.py` | Tách đường kẻ, dựng lưới bảng và ô |
| `img2pdf/layout.py` | Ghép tất cả thành trang: khối, cột, chữ đậm, tiêu đề, hình, thứ tự đọc |
| `img2pdf/vn_correct.py` | Sửa dấu tiếng Việt theo ngữ cảnh |
| `img2pdf/fonts.py` | Phông TrueType, ước lượng cỡ chữ khớp độ rộng dòng gốc |
| `img2pdf/render.py` | Vẽ PDF bằng ReportLab (chữ thật, phông nhúng, bookmark) |
| `img2pdf/gui/` | Giao diện PyQt6 |
| `models/tessdata` | Model Tesseract `vie`, `eng`, `osd` (tessdata_best) |
| `models/dict` | Từ điển Viet74K, `extra_phrases.txt` (cụm hành chính), `user_phrases.txt` (tự thêm) |

Muốn bộ sửa dấu nhận thêm thuật ngữ riêng của cơ quan bạn, tạo `models/dict/user_phrases.txt`,
mỗi dòng một cụm từ (ví dụ `sở tài nguyên và môi trường`).

## Giới hạn

- OCR không bao giờ đúng 100%: ảnh mờ, chữ viết tay, phông trang trí sẽ có lỗi. Hãy soát bản xem
  trước và sửa trực tiếp trong ứng dụng trước khi xuất.
- Bảng **không có đường kẻ** được giữ đúng vị trí dạng các khối chữ, không thành lưới ô.
- Chữ nghiêng (italic) chưa được nhận dạng.
- Chữ viết tay (ngày tháng điền tay…) thường bị đọc sai; hãy sửa trong ứng dụng hoặc bật
  "Giữ nền và màu gốc".
- Một số cặp từ đều đúng chính tả nhưng khác nghĩa (vd. `cấp tỉnh` / `cấp tính`) không thể tự phân
  biệt; bộ sửa dấu chọn giữ nguyên kết quả OCR trong trường hợp này.

## Giấy phép thành phần

Tesseract OCR và tessdata (Apache 2.0), Viet74K (duyet/vietnamese-wordlist), ReportLab (BSD),
PyMuPDF (AGPL), Qt/PyQt6 (GPL), OpenCV (Apache 2.0).
