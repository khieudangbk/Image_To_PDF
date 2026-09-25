# Image To PDF VN

Ứng dụng desktop chuyển **ảnh chụp / ảnh scan tài liệu** thành **file PDF văn bản thật**, giống như khi
xuất từ Word sang PDF. Trong file PDF, chữ là chữ thật (bôi đen, sao chép, tìm kiếm được) với phông chữ
nhúng, chứ không phải ảnh dán vào trang.

## Tính năng

- **OCR tiếng Việt** bằng Tesseract 5 (có thể chọn Việt + Anh).
- **Tự xử lý ảnh chụp**: cắt đúng phần giấy, nắn phẳng ảnh chụp nghiêng hoặc chụp chéo, tự xoay đúng
  chiều, khử bóng và ánh sáng không đều. Giấy tờ có khung được nắn thành hình chữ nhật chuẩn.
- **Dựng lại bố cục**: tiêu đề, đoạn văn, nhiều cột, bảng có đường kẻ, chữ đậm, chữ màu; mỗi dòng
  được đặt đúng vị trí và cỡ chữ như bản gốc. Tiêu đề thành mục lục (bookmark) trong PDF.
- **Giữ hình**: logo, con dấu, chữ ký, ảnh chân dung và chữ viết tay được giữ dạng ảnh.
- **Nền trang**: tự giữ nền gốc (hoa văn, màu giấy) cho giấy tờ có trang trí, còn tài liệu thường
  ra nền trắng sạch. Có thể chọn trong tab Cài đặt.
- **Tự sửa lỗi OCR** bằng dữ liệu tiếng Việt: từ điển từ ghép, địa danh hành chính, chữ viết tắt,
  họ người.
- **Chỉnh sửa trước khi xuất**: bấm vào một vùng để sửa chữ, cỡ chữ, in đậm, căn lề, hoặc xoá vùng.
- Nhiều trang: kéo thả ảnh/thư mục, sắp xếp thứ tự, xử lý song song.
- Xuất **PDF** (A4, Letter hoặc theo khổ ảnh; nhiều phông chữ) và **TXT**.

## Sử dụng bản đóng gói (không cần cài Python)

1. Giải nén `ImageToPDF-win64.zip` (giữ nguyên cả thư mục).
2. Chạy `ImageToPDF\ImageToPDF.exe`.
3. Kéo thả ảnh vào cửa sổ → bấm **Nhận dạng** (F5) → kiểm tra/sửa → **Xuất PDF** (Ctrl+E).

Xử lý hàng loạt bằng dòng lệnh:

```
ImageToPDF.exe --cli thu_muc_anh -o ket_qua.pdf
```

## Chạy từ mã nguồn

Yêu cầu: Windows 10/11, Python 3.11+, [Tesseract 5](https://github.com/UB-Mannheim/tesseract/wiki)
(model tiếng Việt đã có sẵn trong `models/tessdata`).

```
pip install -r requirements.txt
python app.py                                  # giao diện
python cli.py anh1.jpg anh2.png -o ra.pdf      # dòng lệnh
```

Xem các tuỳ chọn dòng lệnh bằng `python cli.py --help`.

## Đóng gói

```
powershell -ExecutionPolicy Bypass -File packaging\build.ps1
```

Tạo `dist\ImageToPDF\ImageToPDF.exe` và `dist\ImageToPDF-win64.zip`.

## Kiểm thử

Các bài kiểm thử nằm trong thư mục `tests/` (độ chính xác OCR, sửa lỗi tiếng Việt, xoay/nắn ảnh,
độ dễ đọc của PDF, giao diện). Tạo ảnh mẫu bằng `python tests\make_samples.py` rồi chạy từng file
`tests\test_*.py` hoặc `tests\evaluate.py`.

## Cấu trúc

| Thư mục/tệp | Vai trò |
|---|---|
| `img2pdf/preprocess.py` | Cắt, nắn, xoay, khử nền ảnh |
| `img2pdf/ocr.py` | Gọi Tesseract |
| `img2pdf/tables.py` | Nhận dạng bảng |
| `img2pdf/layout.py` | Dựng bố cục trang |
| `img2pdf/vn_correct.py` | Sửa lỗi tiếng Việt |
| `img2pdf/restore.py` | Cân sáng, phục hồi màu nền |
| `img2pdf/render.py` | Vẽ PDF |
| `img2pdf/gui/` | Giao diện |
| `models/` | Model OCR và từ điển |

Có thể thêm thuật ngữ riêng vào `models/dict/user_phrases.txt` (mỗi dòng một cụm từ).

## Giới hạn

- OCR không bao giờ đúng 100%; hãy soát bản xem trước và sửa trong ứng dụng trước khi xuất.
- Bảng không có đường kẻ được giữ đúng vị trí nhưng không thành lưới ô.
- Chữ viết tay không được chuyển thành chữ gõ, chỉ giữ dạng ảnh.

## Giấy phép thành phần

Tesseract OCR (Apache 2.0), Viet74K (duyet/vietnamese-wordlist), dvhcvn (daohoangson/dvhcvn),
ReportLab (BSD), PyMuPDF (AGPL), Qt/PyQt6 (GPL), OpenCV (Apache 2.0).
