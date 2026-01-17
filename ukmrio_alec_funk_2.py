# This file contains IO functions (except balancing)
import pandas as pd
import numpy as np
import ukmrio_balancing as balance

# PRE NOWCAST
def make_multipliers(stressor, U, S, Y, yrs, meta):
    """
    Make UKMRIO Multipliers - SUT format
    """
    
    eL_results = np.zeros((meta['Z']['len_idx'], len(yrs)))

    for a, yr in enumerate(yrs):
        # Build Z (transactions matrix)
        Z = np.zeros((meta['Z']['len_idx'], meta['Z']['len_col']))
        Z[meta['bal_use']['rng_idx'], meta['bal_use']['rng_col']] = U[yr]
        Z[meta['bal_sup']['rng_idx'], meta['bal_sup']['rng_col']] = S[yr]

        # Build Y (final demand matrix)
        long_Y = np.zeros((meta['Z']['len_idx'], meta['fd']['len_col']))
        long_Y[meta['bal_fd']['rng_idx'], meta['fd']['rng_col']] = Y[yr]

        # Stressor (e.g., emissions)
        long_stressor = np.zeros(meta['Z']['len_idx'])
        long_stressor[:U[yr].shape[0]] = stressor[yr].iloc[:, 0]

        # Total output
        x = np.sum(Z, axis=1) + np.sum(long_Y, axis=1)
        x[x == 0] = 1e-7  

        # Leontief inverse
        recix = 1 / x
        diag_recix = np.diag(recix)
        L = np.linalg.inv(np.eye(Z.shape[0]) - np.dot(Z, diag_recix))

        # Emission intensity
        intensity = long_stressor * recix

        # eL = emission intensity × Leontief inverse
        eL = np.dot(intensity, L)

        # Store results
        eL_results[:, a] = eL

    # Return as DataFrame with industry index
    eL_df = pd.DataFrame(eL_results, columns=yrs)
    return eL_df

# Step 1
def SUTMRIO_to_SMRIO(S_matrix, U_matrix, Y_matrix, full_labels):
    """
    Convert SUT stuctured UKRMIO into symmetric MRIO format.

    Parameters:
    - S_matrix: Supply matrix (DataFrame or numpy array)
    - U_matrix: Use matrix (DataFrame or numpy array)
    - Y_matrix: Final demand matrix (DataFrame or numpy array)
    - full_labels: list of regional industries

    Returns:
    - B_df: Symmetric intermediate transactions matrix (DataFrame)
    - F_df: Final demand matrix (DataFrame) 
    - VA: Value added vector (numpy array)
    """
    # Convert to numpy arrays if DataFrames
    V = S_matrix.values 
    U = U_matrix.values 
    Y = Y_matrix.values
    h = (U.sum(axis=1) + Y.sum(axis=1)) - V.sum(axis=0)

    V[V == 0] = 1e-7
    U[U == 0] = 1e-7
    Y[Y == 0] = 1e-7

    g = U.sum(axis=0) + h
    q = U.sum(axis=1) + Y.sum(axis=1)

    g_diag_inv = np.diag(1 / g)
    q_diag_inv = np.diag(1 / q)
    
    C = np.dot(U , g_diag_inv)
    D = np.dot(V , q_diag_inv)

    B = np.dot(np.dot(D, C), np.diag(g))
    F = np.dot(D, Y)

    VA = (B.sum(axis=1) + F.sum(axis=1)) - B.sum(axis=0)

    B_df = pd.DataFrame(B, index=full_labels, columns=full_labels)
    F_df = pd.DataFrame(F, index=full_labels)

    return B_df, F_df, VA

def make_simple_foot(yrs, t, y, ghg):
    """
    Calculate simple footprint for all years in yrs.
    Returns a dict: {year: footprint_sum}
    """
    simple_foot = {}

    for year in yrs:
        # Total output
        x = np.sum(t[year], axis=1) + np.sum(y[year], axis=1)
        x[x == 0] = 1e-7  # avoid divide-by-zero

        # Technical coefficients
        big_X = np.tile(np.transpose(x), [1232, 1])
        A = t[year] / big_X

        # Leontief inverse
        I = np.identity(1232)
        invL = I - A
        L = np.linalg.inv(invL)

        # Emissions intensity
        e = ghg[year].to_numpy().ravel() / x

        # Footprint matrix (excluding last FD column)
        bigfoot = np.dot(
            np.dot(np.diag(e), L),
            np.diag(y[year].iloc[:, :-1].sum(axis=1)))

        simple_foot[year] = bigfoot.sum()

    return simple_foot

def make_simple_factors(yrs, t, y, ghg):
    """
    Calculate multipliers from SIOT
    Returns a DataFrame with industries as rows and years as columns.
    """
    results = []

    for year in yrs:
        # Get arrays
        Z = t[year].to_numpy()
        Y = y[year].to_numpy()
        n = Z.shape[0]

        # Total output
        x = Z.sum(axis=1) + Y.sum(axis=1)
        x[x == 0] = 1e-7  # avoid divide-by-zero

        # Technical coefficients A = Z / x_j (column-wise division)
        A = Z / x

        # Leontief inverse
        I = np.eye(n)
        invL = I - A
        L = np.linalg.inv(invL)

        # Emissions intensity e (per unit output)
        g = ghg[year].to_numpy().ravel()
        e = g / x

        # Footprint multipliers (row vector): e * L
        eL = np.dot(e, L)

        results.append(eL)

    # Combine into DataFrame (industries as rows, years as columns)
    df = pd.DataFrame(np.array(results).T, columns=yrs)
    return df


#### Nowcast Functions
def pre_nowcast_func(MRIO, y, countries, row_splits, focus_region="UK", verbose=True):


    # === Step 1: Extract Domestic Blocks ===
    domestic_int = {}
    export_int = {}
    import_int = {}
    value_added_vector = {}
    for country in countries:
        # Use label prefixes to identify each region
        region_labels = [label for label in MRIO.index if label.startswith(country)]
        other_labels = [label for label in MRIO.index if not label.startswith(country)]
        # Calculate intermediate blocks
        domestic_int[country] = MRIO.loc[region_labels, region_labels]
        export_int[country] = MRIO.loc[region_labels, other_labels]
        import_int[country] = MRIO.loc[other_labels, region_labels]
        # Calculate regional VA
        value_added_vector[country] = (MRIO.loc[region_labels].sum(axis=1) + y.loc[region_labels].sum(axis=1) - MRIO.loc[:, region_labels].sum(axis=0))

    # === Step 2: Disaggregate Final Demand ===
    demand_blocks_expanded = {}
    for country in countries:
        # Isolate Y for each regions industry
        region_labels = [label for label in y.index if label.startswith(country)]
        df = y.loc[region_labels].copy()
        last_col = df.iloc[:, -1]
        proportions = row_splits.loc[country]
        # Multiply the last column by each region proportion
        split_columns = {target_region: last_col * proportion
            for target_region, proportion in proportions.items()}
        split_df = pd.DataFrame(split_columns, index=df.index)
        # Aggregate UK columns
        uk_col = df.iloc[:, :-1].sum(axis=1).rename('UK')
        updated_df = pd.concat([uk_col, split_df], axis=1)
        demand_blocks_expanded[country] = updated_df

    disaggregated_y = pd.concat(demand_blocks_expanded.values(), axis=0, ignore_index=False, sort=False)

    # Split into blocks as step 1
    domestic_fd = {}
    export_fd = {}
    import_fd = {}
    for country in countries:
        region_labels = [label for label in disaggregated_y.index if label.startswith(country)]
        other_labels = [label for label in disaggregated_y.index if not label.startswith(country)]

        domestic_fd[country] = disaggregated_y.loc[region_labels, [country]]
        export_fd[country] = disaggregated_y.loc[region_labels, [col for col in disaggregated_y.columns if col != country]]
        import_fd[country] = disaggregated_y.loc[other_labels, [country]]

    # === Step 3: Compute Base Metrics ===
    total_exports = {country: pd.concat([export_int[country], export_fd[country]], axis=1)
        for country in export_int.keys() & export_fd.keys()}

    total_imports = {country: pd.concat([import_int[country], import_fd[country]], axis=1)
        for country in import_int.keys() & import_fd.keys()}

    base_exports = {country: total_exports[country].sum(axis=1) for country in total_exports}
    base_imports = {country: total_imports[country].sum() for country in total_imports}
    base_gdp = {country: value_added_vector[country].sum() for country in value_added_vector}
    base_domestic_demand = {country: domestic_fd[country].sum() + import_fd[country].sum()
        for country in domestic_fd.keys() & import_fd.keys()}

    # === Step 4: Output Balances ===
    output_rows = {}
    output_cols = {}
    balanced_flags = {}
    total_diffs = []

    for country in countries:
        row_sum = (domestic_int[country].sum(axis=1) +
            export_int[country].sum(axis=1) +
            demand_blocks_expanded[country].sum(axis=1))
        col_sum = (domestic_int[country].sum(axis=0) +
            import_int[country].sum(axis=0) +
            value_added_vector[country])

        output_rows[country] = row_sum
        output_cols[country] = col_sum
        is_balanced = np.allclose(row_sum, col_sum, rtol=1e-2, atol=1e-1)
        balanced_flags[country] = is_balanced
        total_diffs.append((row_sum - col_sum).abs().sum())
    if verbose and total_diffs:
        avg_diff = np.mean(total_diffs)
        print(f"\nColumn and Row Sum Balance? Average difference across all regions = {avg_diff:,.2f}")

    # === Step 5: GDP Consistency Check ===
    gdp_va = base_gdp[focus_region]
    gdp_cbop = base_domestic_demand[focus_region].sum() + (base_exports[focus_region].sum() - base_imports[focus_region].sum())

    print(f"\nGDP Consistency Check for {focus_region}")
    print(f"Region GDP (per value added) = {gdp_va:,.2f}")
    print(f"Region GDP (per C + BoP)     = {gdp_cbop:,.2f}")

    return (domestic_int, export_int, import_int, value_added_vector,
    domestic_fd, export_fd, import_fd, demand_blocks_expanded,
    base_exports, base_imports, base_gdp, base_domestic_demand,
    output_rows, output_cols, balanced_flags,
    gdp_va, gdp_cbop)

def project_macro_variables(base_gdp, base_exports, base_imports,
                            gdp_rates, export_rates, import_rates,
                            countries, start_year, lag, focus_region='UK'):
    """
    Projects GDP, exports, and imports forward using growth rates
    applied element-wise to arrays, and prints details of each step.
    """
    start_year = int(start_year)
    rate_years = [str(start_year + s) for s in range(1, lag + 1)]
    
    def project_array(base, rates_df, label):
        out = {}
        for c in countries:
            v = base[c]
            growths = []
            for ry in rate_years:
                if ry in rates_df.columns:
                    g = rates_df.loc[c, ry]
                    v = v * (1 + g)
                    growths.append((ry, g))
            out[c] = v
            if c == focus_region:
                print(f"\n--- {label} projection for {c} ---")
                print(f"Base: {base[c].sum():,.2f}")
                for year, g in growths:
                    print(f"Year {year} growth rate: {g*100:.2f}%")
                print(f"Final projected: {v.sum():,.2f}")
        return out

    gdp_nowcast    = project_array(base_gdp,    gdp_rates,    "GDP")
    export_nowcast = project_array(base_exports, export_rates, "Exports")
    import_nowcast = project_array(base_imports, import_rates, "Imports")

    return gdp_nowcast, export_nowcast, import_nowcast


def project_imports_by_origin(import_int, import_fd, import_rates, countries, start_year, lag):
    """
    Imports by origin = (row-sum(intermediate imports) + first-col(FD imports))
    scaled by cumulative *target-year* import growth.
    """
    start_year = int(start_year)
    rate_years = [str(start_year + s) for s in range(1, lag + 1)]  # target-year columns

    # cumulative factors using target-year columns
    cum = {}
    for c in countries:
        factors = [(1 + import_rates.loc[c, y]) for y in rate_years if y in import_rates.columns]
        cum[c] = float(np.prod(factors)) if factors else 1.0

    imports_nowcast_by_origin = {}
    for c in countries:
        inter = import_int[c].sum(axis=1)
        fd = import_fd[c].iloc[:, 0] if isinstance(import_fd[c], pd.DataFrame) else import_fd[c]
        imports_nowcast_by_origin[c] = (inter + fd) * cum[c]

    return imports_nowcast_by_origin

def compute_base_and_nowcasted_AD(countries, domestic_fd, import_fd, export_fd, export_int, base_gdp, base_exports,
    base_imports, gdp_nowcast, export_nowcast, import_nowcast, focus_region='UK'):
    """
    Compute and compare base and nowcasted demand components (Yrr, EXr, AD) for each region.
    Now includes construction of total_exports from export_int + export_fd.

    Returns:
    - base_yrr, base_exr, base_ad, nowcast_yrr, nowcast_exr, nowcast_ad
    """
    # Build total exports from intermediate + final demand exports
    total_exports = {country: pd.concat([export_int[country], export_fd[country]], axis=1)
        for country in export_int.keys() & export_fd.keys()}

    base_yrr, base_exr, base_ad = {}, {}, {}
    nowcast_yrr, nowcast_exr, nowcast_ad = {}, {}, {}

    # === BASE AD ===
    for country in countries:
        alpha = (import_fd[country].sum() / base_imports[country].sum()).sum()
        alpha = 0 if np.isnan(alpha) else alpha

        gdp = base_gdp[country]
        exports = base_exports[country].sum()
        imports = base_imports[country].sum()
        scalar = gdp - (exports - imports) - (alpha * imports)

        phi = domestic_fd[country] / domestic_fd[country].sum()
        phi = phi.fillna(0)
        base_yrr[country] = phi * scalar

        theta = total_exports[country].div(total_exports[country].sum(axis=0), axis=1)
        base_exr[country] = theta.multiply(total_exports[country].sum(axis=0), axis=1)

        base_ad[country] = base_yrr[country].sum(axis=1) + base_exr[country].sum(axis=1)

    # === NOWCAST AD ===
    for country in countries:
        alpha = (import_fd[country].sum() / base_imports[country].sum()).sum()
        alpha = 0 if np.isnan(alpha) else alpha

        gdp = gdp_nowcast[country]
        exports = export_nowcast[country].sum()
        imports = import_nowcast[country].sum()
        scalar = gdp - (exports - imports) - (alpha * imports)

        phi = domestic_fd[country] / domestic_fd[country].sum()
        phi = phi.fillna(0)
        nowcast_yrr[country] = phi * scalar

        theta = total_exports[country].div(total_exports[country].sum(axis=1), axis=0)
        nowcast_exr[country] = theta.multiply(export_nowcast[country], axis=0)

        nowcast_ad[country] = nowcast_yrr[country].sum(axis=1) + nowcast_exr[country].sum(axis=1)

    # === COMPARATIVE OUTPUT ===
    print(f"\n--- {focus_region} Demand Projection Comparison ---")
    def compare(label, base_val, new_val):
        growth = ((new_val - base_val) / base_val) * 100 if base_val != 0 else float('inf')
        print(f"{label:<30} | Base: {base_val:,.2f} | Nowcast: {new_val:,.2f} | YoY: {growth:.2f}%")

    compare("Yrr (domestic demand)", base_yrr[focus_region].sum().sum(), nowcast_yrr[focus_region].sum().sum())
    compare("EXr (foreign demand)", base_exr[focus_region].sum().sum(), nowcast_exr[focus_region].sum().sum())
    compare("ADr (aggregate demand)", base_ad[focus_region].sum(), nowcast_ad[focus_region].sum())

    print("\n--- Internal Consistency Checks (Does SPIN Base AD = Calculated AD) ---")
    print(f"Check base Yrr  : {domestic_fd[focus_region].sum().sum():,.2f}")
    print(f"Check base EXr  : {total_exports[focus_region].sum().sum():,.2f}")
    print(f"Check base ADr  : {domestic_fd[focus_region].sum().sum() + total_exports[focus_region].sum().sum():,.2f}")

    return base_yrr, base_exr, base_ad, nowcast_yrr, nowcast_exr, nowcast_ad

def compute_output_from_ad(domestic_int, aggregate_demand, output_rows, countries):
    """
    Compute output levels from aggregate demand using the Leontief inverse.

    Parameters:
    - domestic_int: dict of domestic intermediate transaction matrices by region
    - aggregate_demand: dict of aggregate demand vectors (e.g., base_ad or nowcast_ad)
    - output_rows: dict of current output vectors per region (used for technical coefficients)
    - countries: list of regions to process

    Returns:
    - output_estimates: dict of output Series per region
    """
    I = np.identity(len(output_rows[countries[0]]))  # assumes 112 rows
    output_estimates = {}

    for country in countries:
        x = output_rows[country]

        # Technical coefficients matrix A
        output_diag_inv = np.linalg.pinv(np.diag(x.to_numpy()))
        A = domestic_int[country] @ output_diag_inv

        # Leontief inverse
        L = np.linalg.pinv(I - A)

        # Implied output from aggregate demand
        x_hat = L @ aggregate_demand[country]

        output_estimates[country] = pd.Series(x_hat, index=x.index)

    return output_estimates

def build_base_domestic_io_and_constraints(countries, domestic_int, import_int, value_added_vector,
    domestic_fd, import_fd, base_output, base_gdp, base_yrr, export_int, export_fd):
    """
    Build base domestic IO tables and compute row/column constraints.

    Returns:
    - base_domestic_io: full IO table per region
    - base_row_constraint: row totals including imports and value added
    - base_col_constraint: column totals including exports and final demand
    """

    # Build total exports from intermediate + final demand exports
    total_exports = {country: pd.concat([export_int[country], export_fd[country]], axis=1)
                    for country in export_int.keys() & export_fd.keys()}

    base_domestic_io = {}
    base_row_constraint = {}
    base_col_constraint = {}

    # Step 1: Construct full IO tables
    full_fd = {country: pd.concat([domestic_fd[country], import_fd[country]], axis=0) 
               for country in countries}

    for country in countries:
        # Supply side (stacked vertically)
        dom = domestic_int[country]
        imp = import_int[country]
        va = value_added_vector[country].to_frame().T
        va.index = ['value_added']
        stacked = pd.concat([dom, imp, va], axis=0)

        # Demand side (concatenated horizontally)
        ex = total_exports[country]
        fd = full_fd[country]
        full_table = pd.concat([stacked, ex, fd], axis=1).fillna(0)

        base_domestic_io[country] = full_table

    # Step 2: Row constraints (production, imports, value-added)
    for country in countries:
        x = base_output[country].squeeze()
        imp = import_int[country].sum(axis=1) + import_fd[country].iloc[:, 0].squeeze()
        va_value = base_gdp[country].sum()

        row_vector = pd.concat([x, imp])
        row_vector['value_added'] = va_value

        row_vector = row_vector.apply(lambda v: 0.000001 if (pd.isna(v) or np.isinf(v) or v < 0) else v)
        base_row_constraint[country] = row_vector

    # Step 3: Column constraints (intermediate demand + exports + final demand)
    for country in countries:
        output = base_output[country].squeeze()
        exports_int = export_int[country].sum().squeeze()
        exports_fd = export_fd[country].sum().squeeze()
        fd = base_yrr[country].sum() + import_fd[country].sum()

        col_vector = pd.concat([output, exports_int, exports_fd, fd])
        col_vector = col_vector.apply(lambda v: 0.000001 if pd.isna(v) or np.isinf(v) or v < 0 else v)
        base_col_constraint[country] = col_vector

    diffs = []
    for country in countries:
        row_total = base_row_constraint[country].sum()
        col_total = base_col_constraint[country].sum()
        diffs.append(abs(col_total - row_total))

    avg_diff = np.mean(diffs)
    print(f"\nColumn and Row Sum Balance? Average difference across all regions = {avg_diff:,.2f}")


    return base_domestic_io, base_row_constraint, base_col_constraint

def build_nowcast_constraints(countries,
                              output_nowcast,
                              imports_nowcast_by_origin,
                              gdp_nowcast,
                              nowcast_exr,
                              nowcast_yrr,
                              import_nowcast):
    """
    Constructs row and column constraints for the nowcasted IO table.

    Parameters:
    - countries: list of country codes
    - output_nowcast: dict of Series or DataFrames with nowcasted sectoral output
    - import_nowcast_by_origin: dict of Series/DataFrames (sectoral imports by origin)
    - gdp_nowcast: dict of scalar GDP values or Series per country
    - nowcast_exr: dict of DataFrames (export demand per sector)
    - nowcast_yrr: dict of DataFrames (domestic final demand per sector)
    - import_nowcast: dict of DataFrames (sectoral imports from other regions)

    Returns:
    - nowcast_row_constraint: dict of Series (combined output + imports + VA)
    - nowcast_col_constraint: dict of Series (combined uses: output, exports, final demand)
    """
    nowcast_row_constraint = {}
    nowcast_col_constraint = {}

    for country in countries:

        # === Row constraint ===
        x = output_nowcast[country].squeeze()
        imp = imports_nowcast_by_origin[country].squeeze()
        va_value = gdp_nowcast[country].sum() if hasattr(gdp_nowcast[country], 'sum') else gdp_nowcast[country]

        row_vector = pd.concat([x, imp])
        row_vector['value_added'] = va_value
        row_vector = row_vector.apply(lambda v: 0.000001 if (pd.isna(v) or np.isinf(v) or v < 0) else v)
        nowcast_row_constraint[country] = row_vector

        # === Column constraint ===
        output = output_nowcast[country].squeeze()
        exports = nowcast_exr[country].sum().squeeze()
        fd = nowcast_yrr[country].sum() + import_nowcast[country].iloc[-1].squeeze()

        col_vector = pd.concat([output, exports, fd])
        col_vector = col_vector.apply(lambda v: 0.000001 if pd.isna(v) or np.isinf(v) or v < 0 else v)
        nowcast_col_constraint[country] = col_vector

    diffs = []
    for country in countries:
        row_total = nowcast_row_constraint[country].sum()
        col_total = nowcast_row_constraint[country].sum()
        diffs.append(abs(col_total - row_total))

    avg_diff = np.mean(diffs)
    print(f"Column and Row Sum Balance? Average difference across all regions = {avg_diff:,.2f}")

    return nowcast_row_constraint, nowcast_col_constraint

def build_trade_blocks(base_domestic_io, countries):
    """
    Constructs cleaned per-country intermediate and final demand trade blocks, 
    and combines them into global matrices.

    Parameters:
    - base_domestic_io: dict of dataframes keyed by country
    - countries: list of country names to include

    Returns:
    - base_global_intermediate_trade: intermediate trade block (top-left cleared, final demand dropped)
    - base_global_fd_trade: final demand trade block (value_added dropped, exports zeroed)
    - base_international_trade: combined intermediate and final demand trade matrix
    """
    base_intermediate_trade = {}
    base_fd_trade_block = {}

    for country in countries:
        # Intermediate trade: zero out domestic block, drop final demand columns and value_added row
        df_int = base_domestic_io[country].copy()
        df_int.iloc[:112, :112] = 0
        df_int = df_int.iloc[:, :-11]
        df_int = df_int.drop(index='value_added', errors='ignore')
        base_intermediate_trade[country] = df_int

        # Final demand trade: drop value_added, zero last column, keep FD columns only
        df_fd = base_domestic_io[country].copy()
        df_fd = df_fd.drop(index='value_added', errors='ignore')
        df_fd.iloc[:, -1] = 0
        df_fd = df_fd.iloc[:, 1232:]
        base_fd_trade_block[country] = df_fd

    base_global_intermediate_trade = pd.concat(
        [df.iloc[:112, :] for df in base_intermediate_trade.values()],
        axis=0
    )

    base_global_fd_trade = pd.concat(
        [df.iloc[:112, :] for df in base_fd_trade_block.values()],
        axis=0
    )

    base_international_trade = pd.concat(
        [base_global_intermediate_trade, base_global_fd_trade],
        axis=1
    )

    return base_global_intermediate_trade, base_global_fd_trade, base_international_trade

def build_nowcast_MRIO(balanced_global_trade, balanced_io, countries):
    """
    Builds nowcast outputs:
    - y_nowcast: final demand block (last 11 columns)
    - t_nowcast: intermediate trade matrix (without final demand)
    - va_nowcast: value added vector by sector-country

    Parameters:
    - balanced_global_trade: DataFrame template to fill (copied internally)
    - balanced_io: dict of per-country IO DataFrames
    - countries: list of country codes to process

    Returns:
    - y_nowcast: DataFrame of final demand columns
    - t_nowcast: intermediate trade matrix with domestic blocks inserted
    - va_nowcast: concatenated value-added vector
    """
    t_nowcast = balanced_global_trade.copy()

    for country in countries:
        df = balanced_io[country]

        # Insert domestic intermediate block (112x112)
        block = df.iloc[:112, :112]
        row_labels = df.index[:112]
        col_labels = df.columns[:112]
        t_nowcast.loc[row_labels, col_labels] = block.values

        # Insert domestic final demand column for this country, if present
        if country in df.columns:
            domestic_fd_col = df[country]
            domestic_fd_rows = [idx for idx in df.index[:112] if idx.startswith(country)]
            t_nowcast.loc[domestic_fd_rows, country] = domestic_fd_col.loc[domestic_fd_rows].values

    # Extract final demand (last 11 columns)
    y_nowcast = t_nowcast.iloc[:, -11:]

    # Drop final demand from t_nowcast to isolate intermediate trade
    t_nowcast = t_nowcast.iloc[:, :-11]

    # Build value-added vector
    va_parts = []
    for country in countries:
        df = balanced_io[country]
        if 'value_added' in df.index:
            va_row = df.loc['value_added'].iloc[:112]
            va_row.index = [f"{col}_{country}" for col in df.columns[:112]]
            va_parts.append(va_row)

    va_nowcast = pd.concat(va_parts)

    return y_nowcast, t_nowcast, va_nowcast

### Full method 2 pipline
def nowcasting_pipeline_method2(t, y, countries, row_splits, gdp_rates, export_rates, import_rates, base_year, lag=3, focus_region="UK", verbose=False):
    """
    End-to-end MRIO nowcasting pipeline.

    Parameters
    ----------
    t, y : dict[int -> DataFrame]
        Base-year MRIO blocks keyed by year (e.g., t[2016], y[2016]).
    countries : list[str]
        Countries/regions to include.
    row_splits : dict[int -> any]
        Row-split metadata keyed by year (row_splits[base_year]).
    gdp_rates, export_rates, import_rates : pandas.DataFrame
        Growth rate tables with countries as index and years (str) as columns.
    base_year : int
        Base year for nowcasting.
    lag : int, default 3
        Number of forward years to project (applies multiplicatively year by year).
    focus_region : str, default "UK"
        Region for any region-specific diagnostics used by helper funcs.
    verbose : bool, default False
        If True, prints diagnostic summaries.

    Returns
    -------
    y_nowcast : dict[int -> pandas.DataFrame] or pandas.DataFrame
    t_nowcast : dict[int -> pandas.DataFrame] or pandas.DataFrame
    va_nowcast : dict[int -> pandas.Series] or pandas.Series

    Notes
    -----
    Assumes the following helper functions are available in scope:
      - pre_nowcast_func
      - project_macro_variables
      - project_imports_by_origin
      - compute_base_and_nowcasted_AD
      - compute_output_from_ad
      - build_base_domestic_io_and_constraints
      - build_nowcast_constraints
      - gras_balance
      - build_trade_blocks
      - ras_balance_global
      - build_nowcast_MRIO
    """

    # ---------------------------
    # Step 1: Pre-nowcast setup
    # ---------------------------
    (domestic_int, export_int, import_int, value_added_vector, domestic_fd, export_fd,
        import_fd, demand_blocks_expanded, base_exports, base_imports, base_gdp, base_domestic_demand,
        output_rows, output_cols, balanced_flags, gdp_va, gdp_cbop,) = pre_nowcast_func(
        t[base_year], y[base_year], countries, row_splits[base_year], focus_region, verbose,)

    # ---------------------------
    # Step 2: Project macro totals
    # ---------------------------
    gdp_nowcast, export_nowcast, import_nowcast = project_macro_variables(
        base_gdp, base_exports, base_imports, gdp_rates, export_rates,
        import_rates, countries, start_year=base_year, lag=lag,
        focus_region=focus_region,)

    # ---------------------------
    # Step 3: Project imports by origin
    # ---------------------------
    imports_nowcast_by_origin = project_imports_by_origin(
        import_int, import_fd, import_rates, countries, base_year, lag)

    # ---------------------------
    # Step 4: Build base & nowcast AD
    # ---------------------------
    (base_yrr, base_exr, base_ad, nowcast_yrr, nowcast_exr, nowcast_ad,) = compute_base_and_nowcasted_AD(
        countries, domestic_fd, import_fd, export_fd, export_int, base_gdp, base_exports, base_imports,
        gdp_nowcast, export_nowcast, import_nowcast, focus_region=focus_region,)

    # ---------------------------
    # Step 5: Compute outputs from AD
    # ---------------------------
    base_output = compute_output_from_ad(domestic_int, base_ad, output_rows, countries)
    nowcast_output = compute_output_from_ad(domestic_int, nowcast_ad, output_rows, countries)

    # ---------------------------
    # Step 6: Base IO & constraints
    # ---------------------------
    (base_domestic_io, base_row_constraint, base_col_constraint,) = build_base_domestic_io_and_constraints(
        countries, domestic_int, import_int, value_added_vector, domestic_fd, import_fd, base_output,
        base_gdp, base_yrr, export_int, export_fd,)

    # ---------------------------
    # Step 7: Nowcast constraints
    # ---------------------------
    nowcast_row_constraint, nowcast_col_constraint = build_nowcast_constraints(
        countries, nowcast_output, imports_nowcast_by_origin, gdp_nowcast,
        nowcast_exr, nowcast_yrr, import_nowcast,)

    # ---------------------------
    # Step 8: GRAS balance (once)
    # ---------------------------
    balanced_io = balance.gras_balance(base_domestic_io, nowcast_row_constraint, nowcast_col_constraint, countries)

    # ---------------------------
    # Step 10/11: Trade blocks (base & nowcast)
    # ---------------------------
    _, _, base_international_trade = build_trade_blocks(base_domestic_io, countries)
    _, final_demand_nowcast, nowcast_international_trade = build_trade_blocks(balanced_io, countries)

    # Diagnostics (optional)
    if verbose:
        nowcast_export_constraint = nowcast_international_trade.sum(axis=0)
        nowcast_import_constraint = nowcast_international_trade.sum(axis=1)

        base_export_constraint = base_international_trade.sum(axis=0)
        base_import_constraint = base_international_trade.sum(axis=1)

        print(f"Export / Import Difference - Nowcast: "
            f"{nowcast_import_constraint.sum() - nowcast_export_constraint.sum():.2f}")
        print(f"Export / Import Difference - Base   : "
            f"{base_import_constraint.sum() - base_export_constraint.sum():.2f}")

        export_diff = 100 * ((nowcast_export_constraint.sum() - base_export_constraint.sum())/ base_export_constraint.sum())
        import_diff = 100 * ((nowcast_import_constraint.sum() - base_import_constraint.sum())/ base_import_constraint.sum())
        print(f"Export Change (%): {export_diff:.2f}%")
        print(f"Import Change (%): {import_diff:.2f}%")

    # ---------------------------
    # Step 12: Global RAS balance
    # ---------------------------
    # Use summed row/col constraints from nowcast international trade
    nowcast_export_constraint = nowcast_international_trade.sum(axis=0)
    nowcast_import_constraint = nowcast_international_trade.sum(axis=1)

    balanced_global_trade = balance.ras_balance_global(base_international_trade, nowcast_import_constraint,nowcast_export_constraint,)

    if verbose:
        # Totals: Nowcasted (after adjustment)
        total_row_target = nowcast_export_constraint.sum()
        total_col_target = nowcast_import_constraint.sum()
        total_balanced = balanced_global_trade.sum().sum()
        # Totals: Base year
        base_export_constraint = base_international_trade.sum(axis=0)
        base_import_constraint = base_international_trade.sum(axis=1)
        total_row_base = base_export_constraint.sum()
        total_col_base = base_import_constraint.sum()
        total_balanced_base = base_international_trade.sum().sum()
        # Differences (% relative to base)
        row_diff_pct = 100 * (total_row_target - total_row_base) / total_row_base
        col_diff_pct = 100 * (total_col_target - total_col_base) / total_col_base
        balanced_diff_pct = 100 * (total_balanced - total_balanced_base) / total_balanced_base

        print("\nBase Year Totals:")
        print(f"Row constraint total       : {total_row_base:,.6f}")
        print(f"Col constraint total       : {total_col_base:,.6f}")
        print(f"Balanced matrix total      : {total_balanced_base:,.6f}")
        print("Nowcast Totals:")
        print(f"Row constraint total       : {total_row_target:,.6f} ({row_diff_pct:+.2f}%)")
        print(f"Col constraint total       : {total_col_target:,.6f} ({col_diff_pct:+.2f}%)")
        print(f"Balanced matrix total      : {total_balanced:,.6f} ({balanced_diff_pct:+.2f}%)")

    # ---------------------------
    # Step 13: Build nowcasted MRIO
    # ---------------------------
    y_nowcast, t_nowcast, va_nowcast = build_nowcast_MRIO(balanced_global_trade, balanced_io, countries)

    # Return only the three requested outputs
    return y_nowcast, t_nowcast, va_nowcast


### Method 3 functions
def compute_base_and_nowcasted_AD_method3(
    countries,
    domestic_fd, import_fd, export_fd, export_int,
    base_gdp, base_exports, base_imports,
    gdp_nowcast, export_nowcast, import_nowcast,
    contrib_gdp, start_year,
    focus_region='UK'
):
    """
    Compute and compare base and nowcasted demand components (Yrr, EXr, AD) for each region.
    - Builds total_exports = export_int + export_fd
    - For UK ONLY:
        * nowcast_yrr = base_scalar*phi + (nowcast_scalar - base_scalar)*c
        * nowcast_exr uses an industry export vector adjusted by c, then allocates with theta
          where c = contrib_gdp[start_year + 3] (industry-indexed, normalised to sum=1).
    - All other countries use the original method (phi * scalar and theta * export_nowcast).

    Returns:
      base_yrr, base_exr, base_ad, nowcast_yrr, nowcast_exr, nowcast_ad
    """
    import numpy as np
    import pandas as pd

    # --- build total exports matrix (rows=industries, cols=destinations) ---
    total_exports = {
        country: pd.concat([export_int[country], export_fd[country]], axis=1)
        for country in export_int.keys() & export_fd.keys()
    }

    base_yrr, base_exr, base_ad = {}, {}, {}
    nowcast_yrr, nowcast_exr, nowcast_ad = {}, {}, {}

    # === BASE AD ===
    for country in countries:
        # alpha (from BASE year)
        alpha = (import_fd[country].sum() / base_imports[country].sum()).sum()
        alpha = 0 if np.isnan(alpha) else float(alpha)

        gdp = float(base_gdp[country])
        exports = float(base_exports[country].sum())
        imports = float(base_imports[country].sum())
        base_scalar = gdp - (exports - imports) - (alpha * imports)

        # Yrr base
        phi = domestic_fd[country] / domestic_fd[country].sum()
        phi = phi.fillna(0.0)
        base_yrr[country] = phi * base_scalar

        # EXr base (reconstruct from base totals; identity)
        theta_colsums = total_exports[country].sum(axis=0).replace(0, np.nan)
        theta = total_exports[country].div(theta_colsums, axis=1).fillna(0.0)
        base_exr[country] = theta.multiply(total_exports[country].sum(axis=0), axis=1)

        # AD base (industry-level)
        base_ad[country] = base_yrr[country].sum(axis=1) + base_exr[country].sum(axis=1)

    # === NOWCAST AD ===
    target_year = int(start_year) + 3  # e.g., 2014 -> 2017
    if target_year not in contrib_gdp.columns:
        # pick nearest available year if exact column missing
        nearest = min(contrib_gdp.columns, key=lambda y: abs(y - target_year))
        target_year = nearest

    for country in countries:
        # alpha (from BASE year, same as above)
        alpha = (import_fd[country].sum() / base_imports[country].sum()).sum()
        alpha = 0 if np.isnan(alpha) else float(alpha)

        # nowcast totals
        gdp_now = float(gdp_nowcast[country])
        exp_now = float(export_nowcast[country].sum())
        imp_now = float(import_nowcast[country].sum())
        nowcast_scalar = gdp_now - (exp_now - imp_now) - (alpha * imp_now)

        # phi (shares for domestic demand)
        phi = domestic_fd[country] / domestic_fd[country].sum()
        phi = phi.fillna(0.0)

        # theta (row-normalised: sector -> destinations)
        theta_rowsums = total_exports[country].sum(axis=1).replace(0, np.nan)
        theta = total_exports[country].div(theta_rowsums, axis=0).fillna(0.0)

        if country == 'UK':
            # --- UK ONLY: tilt Yrr using contrib vector c ---
            gdp_base = float(base_gdp[country])
            exp_base = float(base_exports[country].sum())
            imp_base = float(base_imports[country].sum())
            base_scalar_here = gdp_base - (exp_base - imp_base) - (alpha * imp_base)
            total_change = nowcast_scalar - base_scalar_here

            # contributions c (industry-indexed), sum exactly to 1, aligned to phi index
            c = (
                pd.to_numeric(contrib_gdp[target_year], errors='coerce')
                .reindex(phi.index)
                .fillna(0.0)
            )
            c = c / (c.sum() or 1.0)

            # ensure shapes match phi (assume single column, e.g. 'UK')
            colname = phi.columns[0] if hasattr(phi, "columns") and len(phi.columns) else country
            c_df = c.to_frame(colname)

            # Yrr: base split + allocated change
            nowcast_yrr[country] = base_scalar_here * phi + total_change * c_df

            # --- UK ONLY: tilt industry export vector by c, then allocate with theta ---
            ex0_by_ind = total_exports[country].sum(axis=1)  # base exports by industry
            E0 = float(ex0_by_ind.sum())
            E1 = float(export_nowcast[country].reindex(ex0_by_ind.index).sum())
            dE = E1 - E0

            ex_by_ind_adj = ex0_by_ind + dE * c
            ex_by_ind_adj *= (E1 / (ex_by_ind_adj.sum() or 1.0))  # hit E1 exactly

            nowcast_exr[country] = theta.multiply(ex_by_ind_adj, axis=0)

        else:
            # --- all OTHER countries: original method ---
            nowcast_yrr[country] = phi * nowcast_scalar
            nowcast_exr[country] = theta.multiply(
                export_nowcast[country].reindex(theta.index), axis=0
            )

        # AD nowcast (industry-level)
        nowcast_ad[country] = nowcast_yrr[country].sum(axis=1) + nowcast_exr[country].sum(axis=1)

    # === COMPARATIVE OUTPUT (focus region) ===
    print(f"\n--- {focus_region} Demand Projection Comparison ---")
    def compare(label, base_val, new_val):
        growth = ((new_val - base_val) / base_val) * 100 if base_val != 0 else float('inf')
        print(f"{label:<30} | Base: {base_val:,.2f} | Nowcast: {new_val:,.2f} | YoY: {growth:.2f}%")

    compare("Yrr (domestic demand)",
            base_yrr[focus_region].sum().sum(),
            nowcast_yrr[focus_region].sum().sum())
    compare("EXr (foreign demand)",
            base_exr[focus_region].sum().sum(),
            nowcast_exr[focus_region].sum().sum())
    compare("ADr (aggregate demand)",
            base_ad[focus_region].sum(),
            nowcast_ad[focus_region].sum())

    print("\n--- Internal Consistency Checks (Does SPIN Base AD = Calculated AD) ---")
    print(f"Check base Yrr  : {domestic_fd[focus_region].sum().sum():,.2f}")
    print(f"Check base EXr  : {total_exports[focus_region].sum().sum():,.2f}")
    print(f"Check base ADr  : {(domestic_fd[focus_region].sum().sum() + total_exports[focus_region].sum().sum()):,.2f}")

    return base_yrr, base_exr, base_ad, nowcast_yrr, nowcast_exr, nowcast_ad


def nowcasting_pipeline_method3(t, y, countries, row_splits, gdp_rates, export_rates, import_rates, contrib_gdp, base_year, lag=3, focus_region="UK", verbose=False):
    """
    End-to-end MRIO nowcasting pipeline.

    Parameters
    ----------
    t, y : dict[int -> DataFrame]
        Base-year MRIO blocks keyed by year (e.g., t[2016], y[2016]).
    countries : list[str]
        Countries/regions to include.
    row_splits : dict[int -> any]
        Row-split metadata keyed by year (row_splits[base_year]).
    gdp_rates, export_rates, import_rates : pandas.DataFrame
        Growth rate tables with countries as index and years (str) as columns.
    base_year : int
        Base year for nowcasting.
    lag : int, default 3
        Number of forward years to project (applies multiplicatively year by year).
    focus_region : str, default "UK"
        Region for any region-specific diagnostics used by helper funcs.
    verbose : bool, default False
        If True, prints diagnostic summaries.

    Returns
    -------
    y_nowcast : dict[int -> pandas.DataFrame] or pandas.DataFrame
    t_nowcast : dict[int -> pandas.DataFrame] or pandas.DataFrame
    va_nowcast : dict[int -> pandas.Series] or pandas.Series

    Notes
    -----
    Assumes the following helper functions are available in scope:
      - pre_nowcast_func
      - project_macro_variables
      - project_imports_by_origin
      - compute_base_and_nowcasted_AD
      - compute_output_from_ad
      - build_base_domestic_io_and_constraints
      - build_nowcast_constraints
      - gras_balance
      - build_trade_blocks
      - ras_balance_global
      - build_nowcast_MRIO
    """

    # ---------------------------
    # Step 1: Pre-nowcast setup
    # ---------------------------
    (domestic_int, export_int, import_int, value_added_vector, domestic_fd, export_fd,
        import_fd, demand_blocks_expanded, base_exports, base_imports, base_gdp, base_domestic_demand,
        output_rows, output_cols, balanced_flags, gdp_va, gdp_cbop,) = pre_nowcast_func(
        t[base_year], y[base_year], countries, row_splits[base_year], focus_region, verbose,)

    # ---------------------------
    # Step 2: Project macro totals
    # ---------------------------
    gdp_nowcast, export_nowcast, import_nowcast = project_macro_variables(
        base_gdp, base_exports, base_imports, gdp_rates, export_rates,
        import_rates, countries, start_year=base_year, lag=lag,
        focus_region=focus_region,)

    # ---------------------------
    # Step 3: Project imports by origin
    # ---------------------------
    imports_nowcast_by_origin = project_imports_by_origin(
        import_int, import_fd, import_rates, countries, base_year, lag)

    # ---------------------------
    # Step 4: Build base & nowcast AD
    # ---------------------------
    (base_yrr, base_exr, base_ad, nowcast_yrr, nowcast_exr, nowcast_ad,) = compute_base_and_nowcasted_AD_method3(
        countries, domestic_fd, import_fd, export_fd, export_int, base_gdp, base_exports,
        base_imports, gdp_nowcast, export_nowcast, import_nowcast, contrib_gdp, start_year=base_year, focus_region=focus_region)

    # ---------------------------
    # Step 5: Compute outputs from AD
    # ---------------------------
    base_output = compute_output_from_ad(domestic_int, base_ad, output_rows, countries)
    nowcast_output = compute_output_from_ad(domestic_int, nowcast_ad, output_rows, countries)

    # ---------------------------
    # Step 6: Base IO & constraints
    # ---------------------------
    (base_domestic_io, base_row_constraint, base_col_constraint,) = build_base_domestic_io_and_constraints(
        countries, domestic_int, import_int, value_added_vector, domestic_fd, import_fd, base_output,
        base_gdp, base_yrr, export_int, export_fd,)

    # ---------------------------
    # Step 7: Nowcast constraints
    # ---------------------------
    nowcast_row_constraint, nowcast_col_constraint = build_nowcast_constraints(
        countries, nowcast_output, imports_nowcast_by_origin, gdp_nowcast,
        nowcast_exr, nowcast_yrr, import_nowcast,)

    # ---------------------------
    # Step 8: GRAS balance (once)
    # ---------------------------
    balanced_io = balance.gras_balance(base_domestic_io, nowcast_row_constraint, nowcast_col_constraint, countries)

    # ---------------------------
    # Step 10/11: Trade blocks (base & nowcast)
    # ---------------------------
    _, _, base_international_trade = build_trade_blocks(base_domestic_io, countries)
    _, final_demand_nowcast, nowcast_international_trade = build_trade_blocks(balanced_io, countries)

    # Diagnostics (optional)
    if verbose:
        nowcast_export_constraint = nowcast_international_trade.sum(axis=0)
        nowcast_import_constraint = nowcast_international_trade.sum(axis=1)

        base_export_constraint = base_international_trade.sum(axis=0)
        base_import_constraint = base_international_trade.sum(axis=1)

        print(f"Export / Import Difference - Nowcast: "
            f"{nowcast_import_constraint.sum() - nowcast_export_constraint.sum():.2f}")
        print(f"Export / Import Difference - Base   : "
            f"{base_import_constraint.sum() - base_export_constraint.sum():.2f}")

        export_diff = 100 * (
            (nowcast_export_constraint.sum() - base_export_constraint.sum())
            / base_export_constraint.sum())
        import_diff = 100 * (
            (nowcast_import_constraint.sum() - base_import_constraint.sum())
            / base_import_constraint.sum())
        print(f"Export Change (%): {export_diff:.2f}%")
        print(f"Import Change (%): {import_diff:.2f}%")

    # ---------------------------
    # Step 12: Global RAS balance
    # ---------------------------
    # Use summed row/col constraints from nowcast international trade
    nowcast_export_constraint = nowcast_international_trade.sum(axis=0)
    nowcast_import_constraint = nowcast_international_trade.sum(axis=1)

    balanced_global_trade = balance.ras_balance_global(base_international_trade, nowcast_import_constraint,nowcast_export_constraint,)

    if verbose:
        # Totals: Nowcasted (after adjustment)
        total_row_target = nowcast_export_constraint.sum()
        total_col_target = nowcast_import_constraint.sum()
        total_balanced = balanced_global_trade.sum().sum()
        # Totals: Base year
        base_export_constraint = base_international_trade.sum(axis=0)
        base_import_constraint = base_international_trade.sum(axis=1)
        total_row_base = base_export_constraint.sum()
        total_col_base = base_import_constraint.sum()
        total_balanced_base = base_international_trade.sum().sum()
        # Differences (% relative to base)
        row_diff_pct = 100 * (total_row_target - total_row_base) / total_row_base
        col_diff_pct = 100 * (total_col_target - total_col_base) / total_col_base
        balanced_diff_pct = 100 * (total_balanced - total_balanced_base) / total_balanced_base

        print("\nBase Year Totals:")
        print(f"Row constraint total       : {total_row_base:,.6f}")
        print(f"Col constraint total       : {total_col_base:,.6f}")
        print(f"Balanced matrix total      : {total_balanced_base:,.6f}")
        print("Nowcast Totals:")
        print(f"Row constraint total       : {total_row_target:,.6f} ({row_diff_pct:+.2f}%)")
        print(f"Col constraint total       : {total_col_target:,.6f} ({col_diff_pct:+.2f}%)")
        print(f"Balanced matrix total      : {total_balanced:,.6f} ({balanced_diff_pct:+.2f}%)")

    # ---------------------------
    # Step 13: Build nowcasted MRIO
    # ---------------------------
    y_nowcast, t_nowcast, va_nowcast = build_nowcast_MRIO(balanced_global_trade, balanced_io, countries)

    # Return only the three requested outputs
    return y_nowcast, t_nowcast, va_nowcast