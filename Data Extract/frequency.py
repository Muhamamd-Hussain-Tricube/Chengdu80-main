from selenium.webdriver import Chrome, ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from keywords import keywords
import requests


def get_frequency(search_value) -> dict[str, int]:
    frequency = dict[str, int]()
    for kw in keywords:
        frequency[kw] = 0
    for v in search_value:
        url = v["url"]
        print(url)
        response = requests.get(url)
        if response.status_code != 200:
            continue

        content = response.text
        for kw in keywords:
            count = content.count(kw)
            if count == 0:
                continue
            frequency[kw] += count
    return frequency
