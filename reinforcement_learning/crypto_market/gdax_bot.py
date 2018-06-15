import requests
import datetime
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import json, hmac, hashlib, time, base64
from requests.auth import AuthBase
import os

np.warnings.filterwarnings('ignore')


def load_setting(setting, file):
    with open(file, 'r') as fd:
        for line in fd.readlines():
            setting_name, setting_value = line.split(':')
            if setting_name == setting:
                return setting_value.strip()
    return None

userdir = os.path.expanduser('~')
SETTINGS_FILE = userdir+'/.gdax_test/settings'

API_KEY = load_setting('API_KEY', SETTINGS_FILE)
API_SECRET = load_setting('API_SECRET', SETTINGS_FILE)
API_PASS = load_setting('API_PASS', SETTINGS_FILE)

WINDOW = 30
MAX_TRIES = 5
HOLD = 0
SELL = 1
BUY = 2
MARKET_PERIOD = 300
TIMEOUT = 5
gdax_test_url = 'https://api-public.sandbox.gdax.com'
gdax_offix_url = 'https://api.gdax.com'
gdax_history_url = gdax_offix_url + '/products/{}/candles'.format('BTC-USD')
gdax_product_ulr = gdax_test_url + '/products/{}/ticker'.format('BTC-USD')
order_id = 'f8e304ac-29a7-4e5c-a158-8f1f3e2276cf'


# Create custom authentication for Exchange
class CoinbaseExchangeAuth(AuthBase):
    def __init__(self, api_key, secret_key, passphrase):
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase

    def __call__(self, request):
        timestamp = str(time.time())
        message = timestamp + request.method + request.path_url + (request.body or '')
        message = message.encode('utf-8')
        hmac_key = base64.b64decode(self.secret_key)
        signature = hmac.new(hmac_key, message, hashlib.sha256)
        signature_b64 = base64.b64encode(signature.digest()).decode('utf-8')

        '''

            CB-ACCESS-KEY The api key as a string.
            CB-ACCESS-SIGN The base64-encoded signature (see Signing a Message).
            CB-ACCESS-TIMESTAMP A timestamp for your request.
            CB-ACCESS-PASSPHRASE The passphrase you specified when creating the API key.

        '''
        request.headers.update({
            'CB-ACCESS-SIGN': signature_b64,
            'CB-ACCESS-TIMESTAMP': timestamp,
            'CB-ACCESS-KEY': self.api_key,
            'CB-ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json'
        })
        return request


def do_sell_action(shares, price, auth):
    order = {
        'size': str(0.5 * shares),
        'price': str(price),
        'side': 'sell',
        'product_id': 'BTC-USD',
    }

    print('Send request for order: ', order)

    r = requests.post(gdax_test_url + '/orders', data=json.dumps(order), auth=auth, timeout=TIMEOUT)
    print(r.url)

    if r.status_code == 200:
        j = r.json()
        print('Sell order accepted:', j)
        return j['id']
    else:
        print('Sell order failed:', r.status_code)

    return None


def do_buy_action(cash, price, auth):
    shares_to_buy = np.round((0.5 * cash) / price, 3)

    if shares_to_buy == 0.:
        print('Not enough money %.2f to buy' % cash)
        return None

    order = {
        'size': str(shares_to_buy),
        'price': str(price),
        'side': 'buy',
        'product_id': 'BTC-USD',
    }

    print('Buy request for order: ', order)

    r = requests.post(gdax_test_url + '/orders', data=json.dumps(order), auth=auth, timeout=TIMEOUT)
    print(r.url)
    if r.status_code == 200:
        j = r.json()
        print('Buy order accepted:', j)
        return j['id']
    else:
        print('Buy order failed: ', r.status_code)

    return None


def has_standing_order(order_id, auth):
    if order_id is None:
        return fetch_order_id(auth)

    print('Sending status request for order:', order_id)

    r = requests.get(gdax_test_url + '/orders/' + str(order_id), auth=auth, timeout=TIMEOUT)
    print(r.url)
    if r.status_code == 200:
        j = r.json()
        print('Order status response:', j)
        if j['status'] == 'done':
            return None, False

        return order_id, True

    elif r.status_code == 404:
        # order cancelled
        print('Received 404 - order cancelled')
        return None, False
    else:
        print('Order status request failed:', r.status_code)
        return order_id, False


def fetch_order_id(auth):
    print("Order ID is None, fetching orders...")
    params = {'status': 'open', 'product_id': 'BTC-USD'}
    r = requests.get(gdax_test_url + '/orders', params=params, auth=auth, timeout=TIMEOUT)
    print(r.url)

    if r.status_code == 200:
        j = r.json()
        print('orders:', j)
        order_id = next((d['id'] for d in j if d['status'] == 'open'), None)
        if order_id is not None:
            print('Found open order: ', order_id)
            return order_id, True
    else:
        print('Failed to fetch orders:', r.status_code)
    return None, False


def cancel_order(order_id, auth):
    r = requests.delete(gdax_test_url + '/orders/' + str(order_id), auth=auth, timeout=TIMEOUT)
    if r.status_code != 200:
        print('Failed to delete order: ', order_id)
        raise Exception('Failed to delete order: ', order_id)


def get_account_balance(auth):
    r = requests.get(gdax_test_url + '/accounts', auth=auth, timeout=TIMEOUT)
    j = r.json()

    shares, cash = None, None

    for d in j:
        if d['currency'] == 'BTC':
            shares = float(d['balance'])
        if d['currency'] == 'USD':
            cash = float(d['balance'])

    return shares, cash


def load_history_data():
    params = {
        'start': (datetime.datetime.now() - datetime.timedelta(minutes=int(5 * 30))).isoformat(),
        'end': datetime.datetime.now().isoformat(),
        'granularity': 300
    }
    request = requests.get(gdax_history_url, params=params, timeout=TIMEOUT)
    history_price_data = None
    if request.status_code == 200:
        j = request.json()
        history_price_data = pd.DataFrame(j, columns=['time', 'low', 'high', 'open', 'close', 'volume'])

    return history_price_data


def load_price_data():
    print('Loading price data...')
    r = requests.get(gdax_product_ulr, timeout=TIMEOUT)
    if r.status_code == 200:
        j = r.json()
        return float(j['price'])

    return None


def compute_input(history_data):
    pct = history_data.pct_change()
    mean = pct.rolling(window=WINDOW).mean().as_matrix()
    median = pct.rolling(window=WINDOW).median().as_matrix()
    std = pct.rolling(window=WINDOW).std().as_matrix()
    upperbb = mean + (2 * std)
    lowerbb = mean - (2 * std)

    return np.array([[pct.as_matrix()[-1],
                      lowerbb[-1],
                      mean[-1],
                      median[-1],
                      upperbb[-1]]])


def compute_action(input):
    clf = joblib.load('models/decision_tree_4.pkl')
    return clf.predict(input)


def main_loop(order_id=None):
    auth = CoinbaseExchangeAuth(API_KEY, API_SECRET, API_PASS)

    order_id, hasOrder = has_standing_order(order_id, auth)
    tries_counter = 0
    while hasOrder and tries_counter < MAX_TRIES:
        print(datetime.datetime.now(), ' - has standing order [', str(order_id), ']')
        time.sleep(300)
        order_id, hasOrder = has_standing_order(order_id, auth)
        tries_counter += 1

    try:
        if hasOrder:
            cancel_order(order_id, auth)
            order_id = None
    except Exception:
        return order_id

    shares, cash = get_account_balance(auth)

    print('Current state:')
    print('\tshares(btc): ', shares)
    print('\tcash: ', cash)

    print('loading history data...')
    history_data = load_history_data()

    if history_data is None:
        print('Failed to load history data')
        return order_id

    history_data.sort_values(by=['time'], inplace=True)
    history_data.reset_index(drop=True, inplace=True)

    history_data = history_data['close']

    print('Loaded %d of history data' % len(history_data))
    price = load_price_data()

    if price is None:
        print('Failed to load price data...')
        return order_id

    history_data.set_value(len(history_data), price)

    print('Current price is: ', price)
    portfolio = float(cash + price * shares)
    print('Portfolio value: ', portfolio)
    ror = (portfolio - 2000.) / 2000.
    print('Ror: ', ror)
    print('Benchmark value:', history_data.pct_change().cumsum().as_matrix()[-1])

    print('Computing action....')

    action = compute_action(compute_input(history_data))

    if action[0] == HOLD:
        print('Chosen action is: HOLD')
    elif action[0] == SELL:
        print('Chosen action is: SELL')
        order_id = do_sell_action(shares, price, auth)
    elif action[0] == BUY:
        print('Chosen action is: BUY')
        order_id = do_buy_action(cash, price, auth)
    else:
        print('NO action chosen: HOLD')

    print('Waiting %d seconds' % MARKET_PERIOD)
    time.sleep(MARKET_PERIOD)
    print('-' * 80)
    return order_id


while True:
    try:
        order_id = main_loop(order_id)
    except Exception as e:
        print(e)
        time.sleep(MARKET_PERIOD)
