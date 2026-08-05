from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


DATASET_ID = "kushagrapandya/visdrone-dataset"


def main() -> None:
    # Thư mục hiện tại nơi bạn chạy lệnh Python
    output_dir = Path.cwd()

    print(f"Đang tải dataset: {DATASET_ID}")
    print(f"Thư mục lưu: {output_dir}")

    api = KaggleApi()
    api.authenticate()

    # Tải và tự động giải nén vào thư mục hiện tại
    api.dataset_download_files(
        dataset=DATASET_ID,
        path=str(output_dir),
        unzip=True,
        quiet=False,
    )

    print(f"\nHoàn tất. Data đã được giải nén tại:")
    print(output_dir)


if __name__ == "__main__":
    main()