import os
import datetime
from io import BytesIO
from datetime import timedelta
import polars as pl
import streamlit as st
import pandas as pd

st.title("Calculateur de bénéfices 💰")


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def init_outdir():
    outdir = './data'
    if not os.path.exists(outdir):
        os.mkdir(outdir)


def optimize_soldes_file(uploaded_file):
    df = pd.read_excel(uploaded_file, parse_dates=["date"])

    df = pl.from_pandas(df)

    df.write_parquet("./data/soldes.parquet")

    return df


def get_clients_ids(df):
    return df["id"].unique()


def generate_clients_daily_profit(clients, start_date, end_date):
    df = pl.DataFrame(
        {col: [] for col in ["id", "date", "solde", "percentage", "profit"]},
        schema=[
            ("id", pl.Int64),
            ("date", pl.Date),
            ("solde", pl.Float64),
            ("percentage", pl.Float64),
            ("profit", pl.Float64),
        ],
    )

    for client in clients:
        solde = 0.0
        client_soldes = (
            pl.scan_parquet("./data/soldes.parquet")
            .filter(pl.col("id") == client)
            .collect()
        )

        for current_date in daterange(start_date, end_date):
            client_solde = client_soldes.filter(pl.col("date") == current_date)

            if client_solde.is_empty():
                df.extend(
                    pl.DataFrame(
                        {
                            "id": [client],
                            "date": [current_date],
                            "solde": [solde],
                            "percentage": [0.0],
                            "profit": [0.0],
                        }
                    )
                )
            else:
                solde = client_solde[0]["solde"][0]

                # if solde < 0:
                #     solde = 0.0

                df.extend(
                    pl.DataFrame(
                        {
                            "id": [client],
                            "date": [current_date],
                            "solde": [solde],
                            "percentage": [0.0],
                            "profit": [0.0],
                        }
                    )
                )

    df.write_parquet("./data/clients_daily_profit.parquet")

    return df


def calculate_clients_daily_percentage(clients_daily_profit, clients, start_date, end_date):
    for current_date in daterange(start_date, end_date):
        current_date_clients = (
            pl.scan_parquet("./data/clients_daily_profit.parquet")
            .filter(date=current_date)
            .collect()
        )
        clients_solde_sum = current_date_clients["solde"].sum()

        if clients_solde_sum:
            for client in clients:
                client_solde = (
                    pl.scan_parquet("./data/clients_daily_profit.parquet")
                    .filter(date=current_date, id=client)
                    .collect()
                )
                if not client_solde.is_empty():
                    clients_daily_profit = clients_daily_profit.with_columns(
                        percentage=pl.when(date=current_date, id=client)
                        .then((client_solde[0]["solde"][0] * 100) / clients_solde_sum)
                        .otherwise("percentage")
                    )

    clients_daily_profit.write_parquet("./data/clients_daily_profit.parquet")

    return clients_daily_profit


def calculate_daily_profit(clients_daily_profit, start_date, end_date):
    daily_profit = pl.DataFrame(
        {col: [] for col in ["date", "solde", "percentage", "profit"]},
        schema=[
            ("date", pl.Date),
            ("solde", pl.Float64),
            ("percentage", pl.Float64),
            ("profit", pl.Float64),
        ],
    )

    soldes_sum = clients_daily_profit["solde"].sum()
    for current_date in daterange(start_date, end_date):
        current_date_soldes_sum = clients_daily_profit.filter(date=current_date)[
            "solde"
        ].sum()

        current_date_soldes_sum_percentage = current_date_soldes_sum * 100 / soldes_sum
        daily_profit.extend(
            pl.DataFrame(
                {
                    "date": [current_date],
                    "solde": [current_date_soldes_sum],
                    "percentage": [current_date_soldes_sum * 100 / soldes_sum],
                    "profit": [current_date_soldes_sum_percentage * profit / 100],
                }
            )
        )

    daily_profit.write_parquet("./data/daily_profit.parquet")

    return daily_profit


def calculate_clients_daily_profit(daily_profit, clients_daily_profit):
    for row in clients_daily_profit.iter_rows(named=True):
        per_day_profit = daily_profit.filter(date=row["date"])[0]["profit"][0]
        clients_daily_profit = clients_daily_profit.with_columns(
            profit=pl.when(id=row["id"], date=row["date"])
            .then(row["percentage"] * per_day_profit / 100)
            .otherwise("profit")
        )

    clients_daily_profit.write_parquet("./data/clients_daily_profit.parquet")

    return clients_daily_profit


def calculate_clients_profit(clients_daily_profit, clients):
    clients_profit = pl.DataFrame(
        {col: [] for col in ["id", "profit"]},
        schema=[
            ("id", pl.Int64),
            ("profit", pl.Float64),
        ],
    )

    for client in clients:
        clients_profit.extend(
            pl.DataFrame(
                {
                    "id": [client],
                    "profit": [clients_daily_profit.filter(id=client)["profit"].sum()],
                }
            )
        )

    clients_profit.write_parquet("./data/clients_profit.parquet")

    return clients_profit


@st.cache_data
def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output) as writer:
        df.to_excel(writer, index=False)
    processed_data = output.getvalue()
    return processed_data


if (
    "last_uploaded_file" not in st.session_state or
    "last_selected_dates" not in st.session_state or
    "last_profit" not in st.session_state or
    "last_soldes" not in st.session_state
):
    st.session_state.last_uploaded_file = None
    st.session_state.last_selected_dates = None
    st.session_state.last_profit = None
    st.session_state.last_soldes = None

with st.container():
    uploaded_file = st.file_uploader(
        "Télécharger le fichier", type=["xlsx"])

    dates = st.date_input(
        label="Sélectionner les dates de début et de fin",
        value=(),
        min_value=datetime.date(2010, 1, 1),
        max_value=datetime.date(2050, 1, 1),
        format="DD/MM/YYYY",
    )

    profit = st.number_input("Profit global")

    if len(dates) == 2 and uploaded_file and profit:
        if (
            st.session_state.last_uploaded_file != uploaded_file or
            st.session_state.last_selected_dates != dates or
            st.session_state.last_profit != profit
        ):
            with st.status("En cours d'exécution... ⏳", expanded=True) as status:
                st.session_state.last_uploaded_file = uploaded_file
                st.session_state.last_selected_dates = dates
                st.session_state.last_profit = profit

                start_date, end_date = dates

                status.update(label="Initier le dossier de données...")
                init_outdir()

                status.update(label="Optimisation des soldes...")
                df = optimize_soldes_file(uploaded_file)

                status.update(label="Extraire les identifiants des clients...")
                clients = get_clients_ids(df)

                status.update(
                    label="Initier les soldes quotidiennes des clients...")
                clients_daily_profit = generate_clients_daily_profit(
                    clients, start_date, end_date)

                status.update(
                    label="Calculer le pourcentage quotidien des clients...")
                clients_daily_profit = calculate_clients_daily_percentage(
                    clients_daily_profit, clients, start_date, end_date)

                status.update(label="Calculer le bénéfice quotidien...")
                daily_profit = calculate_daily_profit(
                    clients_daily_profit, start_date, end_date)

                status.update(
                    label="Calculer le bénéfice quotidien des clients...")
                clients_daily_profit = calculate_clients_daily_profit(
                    daily_profit, clients_daily_profit)

                status.update(label="Calculer les bénéfices des clients...")
                clients_profit = calculate_clients_profit(
                    clients_daily_profit, clients)

                status.update(label="Terminer!", state="complete")

        st.download_button(
            label="Télécharger 💾",
            data=convert_df_to_excel(pd.read_parquet(
                "./data/clients_profit.parquet")),
            file_name="profit.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
