import requests
import logging

def auth_login():
    url="https://api.grupoono.lat/auth"
    payload = {
            'customer_key': 'DEMOMEX',
            'customer_user': 'USERAPI',
            'customer_password': 'y4jAki5od4Sa$oR'
        }
    auth = requests.post(url,data=payload).json()
    if 'status' in auth.keys() and auth['status'] == 'success':
        return {'success':auth['data']}
    else:
        # print(f"ERROR RESPONSE API: {auth['error']}")
        logging.error(f"ERROR RESPONSE API: {auth['error']}")
        return {'error':auth['error']}

def call_api_with_auth(url, payload):
    auth = auth_login()
    if 'success' not in auth:
        return {"error": auth['error']}
    headers = {
        'Authorization': f'Bearer {auth["success"]}',
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    response = requests.post(url, data=payload, headers=headers)
    if response.status_code == 200:
        return response.json()
    return {"error": response.text}
