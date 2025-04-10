from datamodel import Listing, ConversionObservation, Observation, Order, OrderDepth, Trade, TradingState, ProsperityEncoder
from typing import List, Optional, Dict
import string
import json
import jsonpickle


# Create input storage and functions in this class
class Status:

    _position_limits = {
        'RAINFOREST_RESIN': 50,
        'KELP': 50,
    }

    _products = ['RAINFOREST_RESIN', 'KELP']

    def __init__(self, product: str, state: TradingState):
        self.product = product
        self.order_depth = state.order_depths.get(product, None)
        self.position = state.position.get(product, 0)
        self.own_trades = state.own_trades.get(product, [])
        self.market_trades = state.market_trades.get(product, [])
        self.observations = state.observations
        self.timestamp = state.timestamp
        
        # Decode previous trader data to get stored timestamp and other state info
        try:
            decoded_trader_data = jsonpickle.decode(state.traderData) if state.traderData else {}
        except Exception as e:
            decoded_trader_data = {}
        # Retrieve the last known timestamp (if any) for this product.
        self.last_timestamp = decoded_trader_data.get(f"{product}_last_timestamp", None)

        limit = self._position_limits[self.product]
        self.possible_buy_amt = max(0, limit - self.position)
        self.possible_sell_amt = max(0, limit + self.position)

    def best_bid(self):
        best_bid = {}
        for product in self._products:
            best_bid[product] = min(self.order_depth.buy_orders) if self.order_depth and self.order_depth.buy_orders else None
        return best_bid

    def best_ask(self):
        best_ask = {}
        for product in self._products:
            best_ask[product] = min(self.order_depth.sell_orders) if self.order_depth and self.order_depth.sell_orders else None
        return best_ask  

    def bid_ask_spread(self):
        spread = {}
        best_bids = self.best_bid()
        best_asks = self.best_ask()
        for product in self._products:
            if best_bids.get(product) is not None and best_asks.get(product) is not None:
                spread[product] = best_asks[product] - best_bids[product]
            else:
                spread[product] = None  # or another flag value
        return spread

    def mid_price(self):
        if not self.order_depth:
            return None

        buy_orders = self.order_depth.buy_orders
        sell_orders = self.order_depth.sell_orders

        total_value = 0
        total_volume = 0

        for price, volume in buy_orders.items():
            total_value += price * volume
            total_volume += volume

        for price, volume in sell_orders.items():
            total_value += price * abs(volume)
            total_volume += abs(volume)

        if total_volume == 0:
            return None

        return total_value / total_volume
    




class Logic:

    @staticmethod
    def get_possible_quotes(product: str, state: TradingState):
        status = Status(product, state)
        
        best_bid = status.best_bid().get(product)
        best_ask = status.best_ask().get(product)

        possible_bid = None
        possible_ask = None

        # Ensure integer prices and valid spread
        if best_bid is not None and best_ask is not None:
            if best_bid + 1 < best_ask:
                possible_bid = best_bid + 1
                possible_ask = best_ask - 1
        elif best_bid is not None:
            possible_bid = best_bid + 1
        elif best_ask is not None:
            possible_ask = best_ask - 1

        return {"possible_bid": possible_bid, "possible_ask": possible_ask}
    
    @staticmethod
    def available_volume_for_orders(product: str, position: int) -> Dict[str, int]:
        
        limit = Status._position_limits[product]

        max_buy_qty = limit - position
        max_sell_qty = limit + position


        # Ensure they aren't negative (already over limit)
        return {
            "max_buy": max(0, max_buy_qty),
            "max_sell": max(0, max_sell_qty)
        }

    @staticmethod
    def select_best_ask_fill(
        status: Status,
        order_amount: int = 1,
    ) -> Optional[Order]:
        best_bid = max(status.order_depth.buy_orders) if status.order_depth.buy_orders else None
        best_ask = min(status.order_depth.sell_orders) if status.order_depth.sell_orders else None

        proposed_ask = None
        if best_ask is not None and best_bid is not None:
            if best_bid + 1 < best_ask:
                proposed_ask = best_ask - 1
        elif best_ask is not None:
            proposed_ask = best_ask - 1

        if proposed_ask is None:
            return None

        # Call mid_price() here for the fair price of this product
        fair_price = status.mid_price()
        if fair_price is None:
            return None

        if proposed_ask not in status.order_depth.sell_orders:
            if status.order_depth.buy_orders:
                sorted_bids = sorted(status.order_depth.buy_orders.items(), key=lambda x: -x[0])
                best_bid_price, best_bid_volume = sorted_bids[0]
                ideal_margin = proposed_ask - fair_price
                execution_margin = best_bid_price - fair_price

                if execution_margin >= ideal_margin:
                    quantity = min(best_bid_volume, order_amount, status.possible_sell_amt)
                    return Order(status.product, best_bid_price, -int(quantity))

            quantity = min(order_amount, status.possible_sell_amt)
            return Order(status.product, proposed_ask, -int(quantity))

        return None

    @staticmethod
    def mean_reversion(status: Status, trader_data: dict, order_amount: int = 1):
        """
        Mean reversion strategy function that:
          - Decodes and uses previous price history (with timestamps) from trader_data.
          - Updates a rolling window of mid-price observations.
          - Computes a simple moving average (SMA).
          - Places a buy order if the current mid price is significantly below the SMA
            and a sell order if significantly above.
          - Updates trader_data with the current timestamp.
        """
        product = status.product
        current_timestamp = status.timestamp
        mid_price = status.mid_price()
        if mid_price is None:
            return [], trader_data

        # Use a product-specific key for storing price history as a list of (timestamp, mid_price) tuples.
        history_key = f"{product}_price_history"
        if history_key not in trader_data:
            trader_data[history_key] = []
        price_history = trader_data[history_key]
        
        # Append current observation
        price_history.append((current_timestamp, mid_price))
        # If desired, you could also remove stale entries based on time difference.
        # For now, we keep a fixed number of most recent observations.
        if len(price_history) > 20:
            price_history.pop(0)
        trader_data[history_key] = price_history

        # Calculate SMA from the collected mid prices.
        prices = [price for ts, price in price_history]
        sma = sum(prices) / len(prices)

        # Use a threshold based on 1% of the SMA.
        threshold = 0.01 * sma
        orders = []
        if mid_price < sma - threshold:
            # Signal to buy if current price is significantly below SMA.
            if status.order_depth and status.order_depth.sell_orders:
                best_ask = min(status.order_depth.sell_orders)
                quantity = min(order_amount, status.possible_buy_amt)
                orders.append(Order(product, best_ask, quantity))
        elif mid_price > sma + threshold:
            # Signal to sell if price is significantly above SMA.
            if status.order_depth and status.order_depth.buy_orders:
                best_bid = max(status.order_depth.buy_orders)
                quantity = min(order_amount, status.possible_sell_amt)
                orders.append(Order(product, best_bid, -quantity))

        # Update the last seen timestamp for this product in trader_data.
        trader_data[f"{product}_last_timestamp"] = current_timestamp

        return orders, trader_data


class Trader:

    def run(self, state: TradingState):
        result = {}  # type: Dict[str, List[Order]]
        # Decode traderData from the state using jsonpickle
        try:
            trader_data = jsonpickle.decode(state.traderData) if state.traderData else {}
        except Exception:
            trader_data = {}

        for product in state.order_depths:
            status = Status(product, state)
            orders, trader_data = Logic.mean_reversion(status, trader_data)
            result[product] = orders

        # Encode trader_data to a string and pass along for future state in the next call.
        traderData = jsonpickle.encode(trader_data)
        conversions = 1  # Update this as needed based on conversion logic
        return result, conversions, traderData



#Based on the fact that Lambda is stateless AWS can not guarantee any class or global
#variables will stay in place on subsequent calls. We provide possibility of defining a traderData string 
#value as an opportunity to keep the state details. Any Python variable could be serialised into string
#with jsonpickle library and deserialised on the next call based on TradingState.traderData property.
#Container will not interfere with the content. 