import pandas as pd
from typing import Any, Optional
import numpy as np
from dataclasses import dataclass
import string
import itertools


STARTING_CUSTOMER_COUNT_SCALE = 0.05  # relative to population size
NEW_CUSTOMER_CUOUNT_SCALE = 0.01  # relative to starting customer count


@dataclass
class Customers:
    frame: pd.DataFrame

    CUSTOMER_ID_COL: str = "customer_id"
    STORE_ID_COL: str = "store_id"
    PATIENCE_COL: str = "patience"
    FIRST_DAY_COL: str = "starting_day"
    TRANSACTION_DEMAND_SCALE_COL: str = "demand_scale"
    TRANSACTION_DEMAND_TREATMENT_EFFECT_COL: str = "demand_treatment_effect"
    PATIENCE_TREATMENT_EFFECT_COL: str = "arrival_treatment_effect"

    @classmethod
    def sample_from_store_parameters(
        cls,
        store_parameters: "pd.Series[Any]",
        simulated_days: int,
        rng: np.random.Generator,
    ):
        # get starting populations
        starting_customers = store_parameters[StoreParameters.CITY_POPULATION_COL] * STARTING_CUSTOMER_COUNT_SCALE

        # sample until we get a positive number of customers
        starting_customer_count = -1
        while starting_customer_count < 0:
            starting_customer_count = np.round(rng.normal(starting_customers, starting_customers / 2)).astype(int)

        # get new customer arrivals
        new_daily_customers = np.maximum(
            0,
            np.ceil(
                rng.normal(
                    starting_customer_count * NEW_CUSTOMER_CUOUNT_SCALE,
                    starting_customer_count * NEW_CUSTOMER_CUOUNT_SCALE / 2,
                    simulated_days,
                )
            ),
        ).astype(int)
        first_purchase_day = np.cumsum(np.ones(simulated_days)).astype(int)
        new_customer_first_purchase_days = [
            first_purchase_day[i] for i, count in enumerate(new_daily_customers) for _ in range(count)
        ]
        total_custmers = starting_customer_count + np.sum(new_daily_customers)

        frame = pd.DataFrame(
            {
                cls.CUSTOMER_ID_COL: [
                    "".join(rng.choice(list(string.ascii_uppercase), 10)) for i in range(total_custmers)
                ],
                cls.FIRST_DAY_COL: [0] * starting_customer_count + new_customer_first_purchase_days,
                cls.TRANSACTION_DEMAND_SCALE_COL: np.maximum(
                    0,
                    rng.normal(
                        store_parameters[StoreParameters.TRANSACTION_DEMAND_MEAN_COL],
                        store_parameters[StoreParameters.TRANSACTION_DEMAND_MEAN_COL] / 2,
                        total_custmers,
                    ),
                ),
                cls.TRANSACTION_DEMAND_TREATMENT_EFFECT_COL: rng.normal(
                    store_parameters[StoreParameters.TRANSACTION_DEMAND_TREATMENT_EFFECT_COL],
                    np.abs(store_parameters[StoreParameters.TRANSACTION_DEMAND_TREATMENT_EFFECT_COL]) / 2,
                    total_custmers,
                ),
                cls.PATIENCE_COL: np.maximum(
                    1,
                    rng.normal(
                        store_parameters[StoreParameters.CUSTOMER_PATIENCE_MEAN_COL],
                        np.abs(store_parameters[StoreParameters.CUSTOMER_PATIENCE_MEAN_COL]) / 2,
                        total_custmers,
                    ),
                ),
                cls.PATIENCE_TREATMENT_EFFECT_COL: rng.normal(
                    store_parameters[StoreParameters.CUSTOMER_PATIENCE_TREATMENT_EFFECT_COL],
                    np.abs(store_parameters[StoreParameters.CUSTOMER_PATIENCE_TREATMENT_EFFECT_COL]) / 2,
                    total_custmers,
                ),
            }
        )
        frame[cls.STORE_ID_COL] = store_parameters[StoreParameters.STORE_ID_COL]

        return cls(frame)


@dataclass
class CustomerState:
    frame: pd.DataFrame

    CUSTOMER_ID_COL: str = "customer_id"
    DAYS_SINCE_LAST_VISIT_COL: str = "days_since_last_visit"
    IS_CHURNED_COL: str = "is_churned"

    @classmethod
    def initial_from_customers(
        cls,
        customers: Customers,
        rng: np.random.Generator,
    ):
        customers_frame = customers.frame
        frame = pd.DataFrame(
            {
                cls.CUSTOMER_ID_COL: customers.frame[customers.CUSTOMER_ID_COL],
                cls.DAYS_SINCE_LAST_VISIT_COL: rng.uniform(0, customers_frame[Customers.PATIENCE_COL]),
                cls.IS_CHURNED_COL: False,
            }
        )
        return cls(frame)


@dataclass
class Transactions:
    frame: pd.DataFrame

    TRANSACTION_ID_COL: str = "transaction_id"
    CUSTOMER_ID_COL: str = "customer_id"
    STORE_ID_COL: str = "store_id"
    REGION_COL: str = "region"
    LOCATION_TYPE_COL: str = "location"
    CITY_POPULATION_COL: str = "city_population"
    PURCHASE_QUANTITY_DRAW_COL: str = "demand_draw"
    ARRIVAL_PROB_COL: str = "arrival_prob"
    DAYS_SINCE_LAST_VISIT_COL: str = "days_since_last_visit"
    CUSTOMER_PATIENCE_COL: str = "customer_patience"
    PURCHASE_QUANTITY_COL: str = "purchase_quantity"
    DAY_COL: str = "day"
    IS_TREATED_COL: str = "is_treated"


@dataclass
class Store:
    customers: Customers
    customer_state: CustomerState

    store_parameters: "pd.Series[Any]"

    _day_counter: int = 0

    @classmethod
    def sample_from_store_parameters(
        cls,
        store_parameters: "pd.Series[Any]",
        num_simulated_days: int,
        rng: np.random.Generator,
    ):
        customers = Customers.sample_from_store_parameters(store_parameters, num_simulated_days, rng)
        customer_state = CustomerState.initial_from_customers(customers, rng)
        return cls(
            customers=customers,
            customer_state=customer_state,
            store_parameters=store_parameters,
        )

    def get_simulated_day_transactions(
        self,
        rng: np.random.Generator,
        is_treated: bool = False,
    ) -> pd.DataFrame:
        customer_frame = self.customers.frame
        customer_state_frame = self.customer_state.frame

        # draw arrival probabilities
        arrival_utils = (
            customer_state_frame[CustomerState.DAYS_SINCE_LAST_VISIT_COL] - customer_frame[Customers.PATIENCE_COL]
        )
        arrival_probs = np.e**arrival_utils / (1 + np.e**arrival_utils)
        if is_treated:
            arrival_probs = np.minimum(1, arrival_probs * (1 + customer_frame[Customers.PATIENCE_TREATMENT_EFFECT_COL]))
        arrival_draws = rng.binomial(1, arrival_probs)

        # create demand
        transactions_frame = pd.DataFrame(
            {
                Transactions.TRANSACTION_ID_COL: [
                    "".join(rng.choice(list(string.ascii_uppercase), 10)) for i in range(len(customer_frame))
                ],
                Transactions.CUSTOMER_ID_COL: customer_frame[Customers.CUSTOMER_ID_COL],
                Transactions.STORE_ID_COL: [self.store_parameters[StoreParameters.STORE_ID_COL]] * len(customer_frame),
                Transactions.REGION_COL: [self.store_parameters[StoreParameters.REGION_COL]] * len(customer_frame),
                Transactions.CITY_POPULATION_COL: [self.store_parameters[StoreParameters.CITY_POPULATION_COL]]
                * len(customer_frame),
                Transactions.LOCATION_TYPE_COL: self.store_parameters[StoreParameters.LOCATION_TYPE_COL],
                Transactions.PURCHASE_QUANTITY_DRAW_COL: rng.beta(2, 3, len(customer_frame))
                * customer_frame[Customers.TRANSACTION_DEMAND_SCALE_COL]
                + 1,
                Transactions.ARRIVAL_PROB_COL: arrival_probs,
                Transactions.DAYS_SINCE_LAST_VISIT_COL: customer_state_frame[CustomerState.DAYS_SINCE_LAST_VISIT_COL],
                Transactions.CUSTOMER_PATIENCE_COL: customer_frame[Customers.PATIENCE_COL],
                Transactions.IS_TREATED_COL: [is_treated] * len(customer_frame),
                Transactions.DAY_COL: [self._day_counter] * len(customer_frame),
            }
        )
        if is_treated:
            transactions_frame[Transactions.PURCHASE_QUANTITY_DRAW_COL] = transactions_frame[
                Transactions.PURCHASE_QUANTITY_DRAW_COL
            ] * (1 + customer_frame[Customers.TRANSACTION_DEMAND_TREATMENT_EFFECT_COL])

        transactions_frame[Transactions.PURCHASE_QUANTITY_COL] = np.round(
            transactions_frame[Transactions.PURCHASE_QUANTITY_DRAW_COL]
        )

        # tabulate which customers arrived
        arrived_customers_filter = (
            (customer_frame[Customers.FIRST_DAY_COL] <= self._day_counter)
            & (arrival_draws == 1)
            & ~(customer_state_frame[CustomerState.IS_CHURNED_COL])
        )
        appeared_customers_filter = customer_frame[Customers.FIRST_DAY_COL] <= self._day_counter

        # filter transactions to arrived customers
        transactions_frame = transactions_frame[arrived_customers_filter]

        # increment state
        self.customer_state.frame.loc[appeared_customers_filter, CustomerState.DAYS_SINCE_LAST_VISIT_COL] += 1
        self.customer_state.frame.loc[arrived_customers_filter, CustomerState.DAYS_SINCE_LAST_VISIT_COL] = 0
        self._day_counter += 1

        # churn customers
        unchurned_customers = customer_state_frame[~customer_state_frame[CustomerState.IS_CHURNED_COL]]
        mean_customers_to_churn = len(self.customers.frame[Customers.FIRST_DAY_COL] == 0) * NEW_CUSTOMER_CUOUNT_SCALE
        num_customers_to_churn = np.maximum(0, rng.normal(mean_customers_to_churn, mean_customers_to_churn / 2)).astype(
            int
        )
        churned_customers_idx = np.sort(rng.choice(unchurned_customers.index, num_customers_to_churn, replace=False))
        self.customer_state.frame.iloc[
            churned_customers_idx, customer_state_frame.columns.get_loc(CustomerState.IS_CHURNED_COL)
        ] = True  # type: ignore

        return transactions_frame.reset_index(drop=True)


@dataclass(frozen=True)
class Region:
    name: str
    proportion_in_simulation: float
    proportion_in_treated_group: Optional[float]
    transaction_demand_scale: float = 1

    @classmethod
    def get_dummy_region(cls):
        return cls(name="dummy", proportion_in_simulation=0, proportion_in_treated_group=0)


@dataclass(frozen=True)
class LocationType:
    name: str
    proportion_in_simulation: float
    proportion_in_treated_group: Optional[float]
    transaction_demand_treatment_effect: float = 0

    @classmethod
    def get_dummy_location_type(cls):
        return cls(name="dummy", proportion_in_simulation=0, proportion_in_treated_group=0)


@dataclass
class SimulationConfig:
    num_stores: int
    num_treated_stores: int
    days_in_simulation: int
    treatment_start_day: int
    transaction_demand_mean: float
    transaction_demand_scale: float
    customer_patience_mean: float
    customer_patience_scale: float
    tenure_months_mean: float
    tenure_months_scale: float
    city_population_mean: float
    city_population_scale: float
    customer_patience_treatment_effect: float
    city_population_treatment_effect: float
    regions: tuple[Region, ...]
    location_types: tuple[LocationType, ...]

    @classmethod
    def get_default_skewed_config(cls):
        return cls(
            num_stores=120,
            num_treated_stores=20,
            days_in_simulation=20,
            treatment_start_day=10,
            transaction_demand_mean=10,
            transaction_demand_scale=1,
            customer_patience_mean=10,
            customer_patience_scale=1,
            tenure_months_mean=10,
            tenure_months_scale=5,
            city_population_mean=10000,
            city_population_scale=1000,
            customer_patience_treatment_effect=0.05,
            city_population_treatment_effect=0.05,
            regions=(
                Region(
                    name="east",
                    proportion_in_simulation=0.65,
                    proportion_in_treated_group=0.1,
                    transaction_demand_scale=4,
                ),
                Region(
                    name="west",
                    proportion_in_simulation=0.20,
                    proportion_in_treated_group=0.5,
                    transaction_demand_scale=1,
                ),
                Region(
                    name="south",
                    proportion_in_simulation=0.15,
                    proportion_in_treated_group=0.4,
                    transaction_demand_scale=2,
                ),
            ),
            location_types=(
                LocationType(
                    name="mall",
                    proportion_in_simulation=0.50,
                    proportion_in_treated_group=0.1,
                    transaction_demand_treatment_effect=0.1,
                ),
                LocationType(
                    name="strip_mall",
                    proportion_in_simulation=0.15,
                    proportion_in_treated_group=0.3,
                    transaction_demand_treatment_effect=-0.1,
                ),
                LocationType(
                    name="city_center",
                    proportion_in_simulation=0.20,
                    proportion_in_treated_group=0.3,
                    transaction_demand_treatment_effect=-0.1,
                ),
                LocationType(
                    name="stand_alone",
                    proportion_in_simulation=0.15,
                    proportion_in_treated_group=0.3,
                    transaction_demand_treatment_effect=0.4,
                ),
            ),
        )

    @classmethod
    def get_default_random_config(cls):
        skewed_config = cls.get_default_skewed_config()
        random_regions = [
            Region(
                name=r.name,
                proportion_in_simulation=r.proportion_in_simulation,
                proportion_in_treated_group=None,
                transaction_demand_scale=r.transaction_demand_scale,
            )
            for r in skewed_config.regions
        ]
        random_location_types = [
            LocationType(
                name=l.name,
                proportion_in_simulation=l.proportion_in_simulation,
                proportion_in_treated_group=None,
                transaction_demand_treatment_effect=l.transaction_demand_treatment_effect,
            )
            for l in skewed_config.location_types
        ]
        skewed_config.regions = tuple(random_regions)
        skewed_config.location_types = tuple(random_location_types)
        return skewed_config

    @classmethod
    def get_default_balanced_config(cls):
        skewed_config = cls.get_default_skewed_config()
        balanced_regions = [
            Region(
                name=r.name,
                proportion_in_simulation=r.proportion_in_simulation,
                proportion_in_treated_group=r.proportion_in_simulation,
                transaction_demand_scale=r.transaction_demand_scale,
            )
            for r in skewed_config.regions
        ]
        balanced_location_types = [
            LocationType(
                name=l.name,
                proportion_in_simulation=l.proportion_in_simulation,
                proportion_in_treated_group=l.proportion_in_simulation,
                transaction_demand_treatment_effect=l.transaction_demand_treatment_effect,
            )
            for l in skewed_config.location_types
        ]
        skewed_config.regions = tuple(balanced_regions)
        skewed_config.location_types = tuple(balanced_location_types)
        return skewed_config


@dataclass
class StoreParameters:
    frame: pd.DataFrame

    STORE_ID_COL: str = "store_id"
    REGION_COL: str = "region"
    LOCATION_TYPE_COL: str = "location"
    CITY_POPULATION_COL: str = "city_population_mean"
    TRANSACTION_DEMAND_MEAN_COL: str = "transaction_demand_mean"
    CUSTOMER_PATIENCE_MEAN_COL: str = "customer_patience_mean"
    TENURE_MONTHS_MEAN_COL: str = "tenure_months_mean"
    TRANSACTION_DEMAND_TREATMENT_EFFECT_COL: str = "transaction_demand_treatment_effect"
    CUSTOMER_PATIENCE_TREATMENT_EFFECT_COL: str = "customer_patience_treatment_effect"
    IS_TREATED_COL: str = "is_treated"

    @classmethod
    def sample_store_parameters(
        cls,
        config: SimulationConfig,
        rng: np.random.Generator,
    ):
        # region determines demand scale
        region_demand_scales = {
            r.name: np.maximum(
                0.5,
                rng.normal(
                    config.transaction_demand_mean * r.transaction_demand_scale,
                    config.transaction_demand_scale,
                    1,
                ),
            )
            for r in config.regions
        }
        sampled_regions = rng.choice(
            [i.name for i in config.regions],
            config.num_stores,
            p=[i.proportion_in_simulation for i in config.regions],
        )
        sampled_transaction_demand_means = [
            np.maximum(
                0.5,
                rng.normal(region_demand_scales[i], region_demand_scales[i] / 2),
            )[0]
            for i in sampled_regions
        ]

        # location determines treatment effect
        location_treatment_effects = {i.name: i.transaction_demand_treatment_effect for i in config.location_types}
        sampled_locations = rng.choice(
            [i.name for i in config.location_types],
            config.num_stores,
            p=[i.proportion_in_simulation for i in config.location_types],
        )
        sampled_transaction_demand_treatment_effect_means = [
            rng.normal(location_treatment_effects[i], np.abs(location_treatment_effects[i])) for i in sampled_locations
        ]

        parameter_frame = pd.DataFrame(
            {
                StoreParameters.STORE_ID_COL: range(config.num_stores),
                StoreParameters.REGION_COL: sampled_regions,
                StoreParameters.LOCATION_TYPE_COL: sampled_locations,
                StoreParameters.TRANSACTION_DEMAND_MEAN_COL: sampled_transaction_demand_means,
                StoreParameters.CUSTOMER_PATIENCE_MEAN_COL: np.maximum(
                    1,
                    rng.normal(
                        config.customer_patience_mean,
                        config.customer_patience_scale,
                        config.num_stores,
                    ),
                ),
                StoreParameters.TENURE_MONTHS_MEAN_COL: np.maximum(
                    3,
                    rng.normal(
                        config.tenure_months_mean,
                        config.tenure_months_scale,
                        config.num_stores,
                    ),
                ),
                StoreParameters.CITY_POPULATION_COL: np.maximum(
                    config.city_population_mean * 0.2,
                    rng.normal(
                        config.city_population_mean,
                        config.city_population_scale,
                        config.num_stores,
                    ),
                ),
                StoreParameters.TRANSACTION_DEMAND_TREATMENT_EFFECT_COL: sampled_transaction_demand_treatment_effect_means,
                StoreParameters.CUSTOMER_PATIENCE_TREATMENT_EFFECT_COL: rng.normal(
                    config.customer_patience_treatment_effect,
                    config.customer_patience_treatment_effect,
                    config.num_stores,
                ),
                StoreParameters.IS_TREATED_COL: False,
            }
        )

        # scale patience by city population
        min_population = parameter_frame[StoreParameters.CITY_POPULATION_COL].min()
        max_population = parameter_frame[StoreParameters.CITY_POPULATION_COL].max()
        min_scale = 0.5
        max_scale = 1.5

        parameter_frame[StoreParameters.CUSTOMER_PATIENCE_MEAN_COL] = min_scale + (max_scale - min_scale) * (
            parameter_frame[StoreParameters.CUSTOMER_PATIENCE_MEAN_COL] - min_population
        ) / (max_population - min_population)

        # assign treatments to regions and locations according to proportions
        total_region_treatment_proportion = sum(
            [i.proportion_in_treated_group for i in config.regions if i.proportion_in_treated_group]
        )
        normalized_region_treatment_proportions = {
            i: i.proportion_in_treated_group / total_region_treatment_proportion
            for i in config.regions
            if i.proportion_in_treated_group and i.proportion_in_treated_group > 0
        }
        if not normalized_region_treatment_proportions:
            normalized_region_treatment_proportions = {Region.get_dummy_region(): 1}

        total_location_treatment_proportion = sum(
            [i.proportion_in_treated_group for i in config.location_types if i.proportion_in_treated_group]
        )
        normalized_location_treatment_proportions = {
            i: i.proportion_in_treated_group / total_location_treatment_proportion
            for i in config.location_types
            if i.proportion_in_treated_group and i.proportion_in_treated_group > 0
        }
        if not normalized_location_treatment_proportions:
            normalized_location_treatment_proportions = {LocationType.get_dummy_location_type(): 1}

        num_treated_stores_by_region_and_location = {
            i: np.ceil(
                normalized_region_treatment_proportions[i[0]]
                * normalized_location_treatment_proportions[i[1]]
                * config.num_treated_stores
            ).astype(int)
            for i in itertools.product(
                normalized_region_treatment_proportions.keys(), normalized_location_treatment_proportions.keys()
            )
        }

        for region_location, num_stores_in_treatment in num_treated_stores_by_region_and_location.items():
            region, location = region_location
            if region == Region.get_dummy_region():
                region_filter = pd.Series([True] * len(parameter_frame))
            else:
                region_filter = pd.Series([True] * len(parameter_frame))
            if location == LocationType.get_dummy_location_type():
                location_filter = region_filter = pd.Series([True] * len(parameter_frame))
            else:
                location_filter = pd.Series([True] * len(parameter_frame))
            target_stores = parameter_frame[region_filter & location_filter]
            treated_stores = rng.choice(target_stores.index, num_stores_in_treatment, replace=False)
            parameter_frame.loc[treated_stores, StoreParameters.IS_TREATED_COL] = True

        num_stores_in_treatment_by_region_and_location = [
            (
                np.ceil(
                    config.num_treated_stores * (i.proportion_in_treated_group / total_region_treatment_proportion)
                ).astype(int)
                if i.proportion_in_treated_group and i.proportion_in_treated_group > 0
                else 0
            )
            for i in config.regions
        ]
        for region, num_stores_in_treatment in zip(config.regions, num_stores_in_treatment_by_region_and_location):
            region_stores = parameter_frame[parameter_frame[StoreParameters.REGION_COL] == region.name]
            treated_stores = rng.choice(region_stores.index, num_stores_in_treatment, replace=False)
            parameter_frame.loc[treated_stores, StoreParameters.IS_TREATED_COL] = True

        return cls(parameter_frame)


@dataclass
class Simulator:
    """
    demand = alpha_1 * treatment + alpha_2 * region + alpha_3 * tenure + alpha_4 * treatment * city_population

    treatment operates via 3 mechanisms:
    1) increasing demand per visit
    2) increasing rate of arrival
    3) increasing customers per day

    region operates via increased demand per visit
    customer_demand_per_visit ~ N(region_demand_per_visit, sigma)
    customer_demand = treatment_demand_scale * customer_demand_per_visit

    tenure operates via increased arrival probabilities
    customer_patience ~ N(tenure_patience, sigma)
    arrival_probability = treatment_arrival_scale *
        e^(days_since_last_arrival - customer_patience + day_boost) / (1 + e^(days_since_last_arrival - customer_patience + day_boost))

    city_population operates via increased customers per day
    customers_per_day = N(city_population_customers, sigma)
    customers_per_day = treatment_customers_scale * customers_per_day

    initial_customers ~ N(city_population_customers * tenure, sigma)
    """

    config: SimulationConfig
    store_parameters: StoreParameters
    stores: list[Store]

    @classmethod
    def initialize_from_config(
        cls,
        config: SimulationConfig,
        rng: np.random.Generator,
    ):
        store_parameters = StoreParameters.sample_store_parameters(config, rng)
        stores = [
            Store.sample_from_store_parameters(row, config.days_in_simulation, rng)
            for _, row in store_parameters.frame.iterrows()
        ]
        return cls(config, store_parameters, stores)

    def get_transactions(
        self,
        rng: np.random.Generator,
    ):
        untreated_days = self.config.treatment_start_day
        treated_days = self.config.days_in_simulation - untreated_days

        transactions = []
        for store in self.stores:
            untreated_transactions = [store.get_simulated_day_transactions(rng) for i in range(untreated_days)]
            treated_transactions = [
                store.get_simulated_day_transactions(rng, store.store_parameters[StoreParameters.IS_TREATED_COL])
                for i in range(treated_days)
            ]
            transactions += untreated_transactions + treated_transactions

        transactions_frame = pd.concat(transactions).sort_values(Transactions.DAY_COL).reset_index(drop=True)
        return transactions_frame

    def get_customers(self):
        return pd.concat([store.customers.frame for store in self.stores])


if __name__ == "__main__":
    RNG = np.random.default_rng(0)
    simulation_config = SimulationConfig.get_default_random_config()
    simulator = Simulator.initialize_from_config(simulation_config, RNG)
    transactions = simulator.get_transactions(RNG)
    pass
