import plotly.express as px
import streamlit as st

from utils import SEASONS, TEAL, default_cities, plot, seasonal_summary, select_city

st.title("Seasonal Analysis")
st.caption("Winter: Dec–Feb · Summer: Mar–May · Monsoon: Jun–Sep · Post-Monsoon: Oct–Nov")
data = seasonal_summary()
cities = sorted(data["City"].unique())
selected = st.multiselect("Cities", cities, default=default_cities(cities), key="seasonal_cities")
if not selected:
    st.info("Select at least one city to explore seasonal AQI.")
    st.stop()
data = data[data["City"].isin(selected)].copy()
city = select_city(selected, "seasonal_focus", "City to inspect")
focus = data[data["City"] == city].set_index("Season").reindex(SEASONS)
worst = focus["average_aqi"].max()
worst_seasons = focus.index[focus["average_aqi"] == worst].tolist()
st.metric(f"Highest average season · {city}", ", ".join(worst_seasons), border=True)
st.caption(f"Average AQI: {worst:.1f}. Seasons pool all available years; they describe historical patterns.")
for col, season in zip(st.columns(4), SEASONS):
    value = focus.loc[season, "average_aqi"]
    col.metric(season, f"{value:.1f}" if value == value else "Unavailable", border=True)

plot(px.bar(data, x="City", y="average_aqi", color="Season", barmode="group",
            category_orders={"Season": SEASONS}, hover_data=["valid_days"],
            title="Seasonal AQI by city", labels={"average_aqi": "Average AQI"}))
heat = data.pivot(index="City", columns="Season", values="average_aqi").reindex(columns=SEASONS)
plot(px.imshow(heat, text_auto=".0f", aspect="auto", color_continuous_scale="YlOrRd",
               title="City × Season · average AQI", labels={"color": "Average AQI"},
               height=max(320, 35 * len(heat) + 120)))

# Pool observations, rather than giving unequal city samples equal weight.
data["aqi_sum"] = data["average_aqi"] * data["valid_days"]
pooled = data.groupby("Season")[["aqi_sum", "valid_days"]].sum().reindex(SEASONS)
pooled["Average AQI"] = pooled["aqi_sum"] / pooled["valid_days"]
plot(px.bar(pooled.reset_index(), x="Season", y="Average AQI", hover_data=["valid_days"],
            title="Average AQI by season · selected cities", color_discrete_sequence=[TEAL]))
st.caption("The combined seasonal average weights each validated observation equally. Seasonal coverage and the number of available years differ across cities.")
