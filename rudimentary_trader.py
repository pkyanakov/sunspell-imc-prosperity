from datamodel import Listing, ConversionObservation, Observation, Order, OrderDepth, Trade, TradingState, ProsperityEncoder
from typing import List, Optional, Dict
import string
import json


# Create input storage and functions in this class
class Status:

    _position_limits = {
        'RAINFOREST_RESIN' : 50,
        'KELP' : 50,
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

        limit = self._position_limits[self.product]
        self.possible_buy_amt = max(0, limit - self.position)
        self.possible_sell_amt = max(0, limit + self.position)


    def best_bid(self):
        best_bid = {}
        for product in self._products:
            best_bid[product] = min(self.order_depth.buy_orders)
        return best_bid

    def best_ask(self):
        best_ask = {}
        for product in self._products:
            best_ask[product] = min(self.order_depth.sell_orders)
        return best_ask  
    
    def bid_ask_spread(self):
        spread = {}
        best_bids = self.best_bid()
        best_asks = self.best_ask()

        for product in self._products:
            if product in best_bids and product in best_asks:
                spread[product] = best_asks[product] - best_bids[product]
            else:
                spread[product] = None  # or float('inf') / -1 if you want to flag missing data

        return spread
    
    def mid_price(self):
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
            return None  # or float('inf') / fallback

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



class Trader:

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        

        for product in state.order_depths:
            status = Status(product, state)
            orders: List[Order] = []

            
            fair_price = Status.mid_price

            
            ask_order = Logic.select_best_ask_fill(status)

            if ask_order:
                orders.append(ask_order)

            # Optional: add matching bid logic here

            result[product] = orders

        traderData = "SAMPLE"
        conversions = 1
        return result, conversions, traderData



#Based on the fact that Lambda is stateless AWS can not guarantee any class or global
#variables will stay in place on subsequent calls. We provide possibility of defining a traderData string 
#value as an opportunity to keep the state details. Any Python variable could be serialised into string
#with jsonpickle library and deserialised on the next call based on TradingState.traderData property.
#Container will not interfere with the content. 