# This script will download images and generate tags automatically from provided set of image's urls from Danbooru.
# Require selenium to be installed. You can install it with pip install selenium==4.15.2.
import os

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

import requests

from lightning_accelerate.utils.config_utils import parse_config_from_cli


def main(config):
    input_path = config["urls_path"]
    output_dir = config.get("output_dir", "data/danbooru")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-extensions")
    driver = webdriver.Chrome(options=chrome_options)

    with open(input_path, "r") as f:
        urls = [url.strip() for url in f.readlines()]

    for url in urls:
        img_id = url.split("?")[0].split("/")[-1]
        assert img_id.isdigit(), f"Invalid url {url}"

        print(url)
        driver.get(url)

        ### Get image
        image_xpath = '//*[@id="content"]/div/a[@class="image-view-original-link"]'
        image_url = driver.find_element(By.XPATH, image_xpath).get_attribute("href")
        img_save_path = os.path.join(output_dir, f"{img_id}.png")
        
        if not os.path.exists(img_save_path):
            r = requests.get(image_url)
            with open(img_save_path, "wb") as f:
                f.write(r.content)

        ### Get tags
        tag_save_path = os.path.join(output_dir, f"{img_id}.txt")
        if not os.path.exists(tag_save_path):
            tags = []

            general_tags_xpath = '//*[@id="tag-list"]/div/ul[@class="general-tag-list"]/li'
            general_tags = driver.find_elements(By.XPATH, general_tags_xpath)

            for tag_element in general_tags:
                tag = tag_element.get_attribute("data-tag-name")
                tags.append(tag)
            character_tags_xpath = (
                '//*[@id="tag-list"]/div/ul[@class="character-tag-list"]/li'
            )
            character_tags = driver.find_elements(By.XPATH, character_tags_xpath)

            for tag_element in character_tags:
                tag = tag_element.get_attribute("data-tag-name")
                tags.append(tag)

            with open(tag_save_path, "w") as f:
                f.write(",".join(tags))

    driver.close()


if __name__ == "__main__":
    config = parse_config_from_cli()
    main(config)
