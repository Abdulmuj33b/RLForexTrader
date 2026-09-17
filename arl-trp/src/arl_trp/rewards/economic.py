class EconomicReward:

    @staticmethod
    def calculate(
        equity_before: float,
        equity_after: float,
    ) -> float:

        return equity_after - equity_before
