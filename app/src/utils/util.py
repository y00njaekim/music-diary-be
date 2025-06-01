import requests
import numpy as np


def fetch_json(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
    except requests.exceptions.ConnectionError as conn_err:
        print(f'Error connecting: {conn_err}')
    except requests.exceptions.Timeout as timeout_err:
        print(f'Timeout error occurred: {timeout_err}')
    except requests.exceptions.RequestException as req_err:
        print(f'Error occurred: {req_err}')
    except ValueError as json_err:
        print(f'JSON decode error: {json_err}')


def check_include(start1, end1, start2, end2):
    return start1 <= start2 <= end1 or start1 <= end2 <= end1 or (start1 >= start2 and end1 <= end2)


def normalize(data):
    min_data = np.min(data)
    max_data = np.max(data)
    return (data - min_data) / (max_data - min_data) if max_data > min_data else data


def hz_to_midi(hz):
    return round(69 + 12 * np.log2(hz/440.))