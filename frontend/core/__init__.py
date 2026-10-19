"""
Pure data logic for the dashboard: everything that *calculates*.

Nothing in this package imports Streamlit or builds HTML. Components in
frontend/components take the results and display them. Keeping the two apart
means the logic can be tested with plain DataFrames and reused elsewhere.

  geography   regions, state boxes, region filter, map views
  danger      risk levels and FFDI bands
  summaries   region summary, day options, riskiest / calmest day
  hotspots    places at High/Extreme risk on the most days
  chart_data  per-day series behind the Insights charts
  places      towns, nearest-town descriptions, towns in at-risk areas
  plain       everyday-language numbers ("1 in 8", areas, weather words)
  explain     plain-English reasons for a cell's risk
  types       shapes of the dictionaries passed to components
"""
