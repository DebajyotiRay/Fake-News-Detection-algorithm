from huggingface_hub import upload_folder

upload_folder(
    repo_id="CrypticRAY/fake-news-distilbert",
    folder_path="./model",   # path to your trained model directory
    repo_type="model"
)
