# real_panel.csv.gz

Derived, anonymised extract of a public-domain quarterly financial panel.
Columns: `unit` (anonymised integer id), `period` (A = 2017-2019, B = 2023-2025),
`y` (return ratio, percent), `x1..x8` (ratio features; `x1` is log scale). Pooled
1st/99th percentile winsorising; row order shuffled within unit. 96 units,
1,152 rows per period.
