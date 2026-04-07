"""
Deep data audit against every ingest script contract.
Run: python scripts/audit_data.py
"""
import sqlite3
import json
import re

con = sqlite3.connect("./data/staging/staging.db")
PASS = "  OK "
FAIL = "  FAIL"
WARN = "  WARN"

issues = []


def _json_ok(s):
    try:
        json.loads(s)
        return True
    except Exception:
        return False

def check(label, expr, warn=False):
    tag = PASS if expr else (WARN if warn else FAIL)
    print(f"{tag}  {label}")
    if not expr:
        issues.append((label, warn))


# ===========================================================================
print("=" * 65)
print("AUDIT: ingest_donki_cme -> cme_events")
print("=" * 65)

n_null_pk = con.execute("SELECT COUNT(*) FROM cme_events WHERE activity_id IS NULL").fetchone()[0]
check("activity_id never NULL", n_null_pk == 0)

n_ar_zero = con.execute("SELECT COUNT(*) FROM cme_events WHERE active_region_num = 0").fetchone()[0]
check("active_region_num: 0 never stored (-> None) [RULE-003]", n_ar_zero == 0)

n_bad_ts = con.execute("SELECT COUNT(*) FROM cme_events WHERE start_time IS NOT NULL AND start_time NOT LIKE '%T%'").fetchone()[0]
check("start_time is ISO format (contains T)", n_bad_ts == 0)

n_bad_cat = con.execute("SELECT COUNT(*) FROM cme_events WHERE source_catalog != 'DONKI'").fetchone()[0]
check("source_catalog always 'DONKI'", n_bad_cat == 0)

bad_linked = con.execute(
    "SELECT COUNT(*) FROM cme_events WHERE linked_flare_id IS NOT NULL AND linked_flare_id NOT LIKE '%-FLR-%'"
).fetchone()[0]
check("linked_flare_id only -FLR- pattern when non-null", bad_linked == 0)

rows = con.execute("SELECT activity_id, linked_event_ids FROM cme_events WHERE linked_event_ids IS NOT NULL LIMIT 500").fetchall()
bad_json = sum(1 for _, lev in rows if not _json_ok(lev))
check("linked_event_ids valid JSON when non-null", bad_json == 0)

n_speed_sentinel = con.execute("SELECT COUNT(*) FROM cme_events WHERE speed_kms > 9000").fetchone()[0]
check("speed_kms: no sentinel values > 9000 [RULE-120]", n_speed_sentinel == 0)

n_speed_bad = con.execute(
    "SELECT COUNT(*) FROM cme_events WHERE speed_kms IS NOT NULL AND (speed_kms < 10 OR speed_kms > 5000)"
).fetchone()[0]
check("speed_kms in plausible range 10-5000 km/s when non-null", n_speed_bad == 0, warn=True)

total_cme = con.execute("SELECT COUNT(*) FROM cme_events").fetchone()[0]
print(f"     Total cme_events: {total_cme:,}")


# ===========================================================================
print()
print("=" * 65)
print("AUDIT: ingest_cdaw_lasco -> cdaw_cme_events")
print("=" * 65)

total_cdaw = con.execute("SELECT COUNT(*) FROM cdaw_cme_events").fetchone()[0]

n_null_cdaw = con.execute("SELECT COUNT(*) FROM cdaw_cme_events WHERE cdaw_id IS NULL").fetchone()[0]
check("cdaw_id never NULL", n_null_cdaw == 0)

cdaw_id_re = re.compile(r"^\d{8}\.\d{6}$")
sample_ids = [r[0] for r in con.execute("SELECT cdaw_id FROM cdaw_cme_events LIMIT 500").fetchall()]
bad_ids = [i for i in sample_ids if not cdaw_id_re.match(i)]
check("cdaw_id matches YYYYMMDD.HHMMSS (sample 500)", len(bad_ids) == 0)

n_halo_pa_set = con.execute(
    "SELECT COUNT(*) FROM cdaw_cme_events WHERE angular_width_deg = 360 AND central_pa_deg IS NOT NULL"
).fetchone()[0]
check("Halo: central_pa_deg IS NULL when width=360 [RULE-052]", n_halo_pa_set == 0)

n_speed_sentinel = con.execute("SELECT COUNT(*) FROM cdaw_cme_events WHERE linear_speed_kms > 9000").fetchone()[0]
check("linear_speed_kms: no sentinel values > 9000", n_speed_sentinel == 0)

n_speed20_sentinel = con.execute("SELECT COUNT(*) FROM cdaw_cme_events WHERE speed_20rs_kms > 9000").fetchone()[0]
check("speed_20rs_kms: no sentinel values > 9000 [RULE-053]", n_speed20_sentinel == 0)

bad_qf = con.execute("SELECT COUNT(*) FROM cdaw_cme_events WHERE quality_flag NOT IN (1, 2, 3)").fetchone()[0]
check("quality_flag always 1, 2, or 3", bad_qf == 0)

n_bad_cat_cdaw = con.execute("SELECT COUNT(*) FROM cdaw_cme_events WHERE source_catalog != 'CDAW'").fetchone()[0]
check("source_catalog always 'CDAW'", n_bad_cat_cdaw == 0)

n_bad_dt_cdaw = con.execute(
    "SELECT COUNT(*) FROM cdaw_cme_events WHERE datetime IS NOT NULL AND datetime NOT LIKE '%T%'"
).fetchone()[0]
check("datetime column has ISO T separator", n_bad_dt_cdaw == 0)

n_mass_null = con.execute("SELECT COUNT(*) FROM cdaw_cme_events WHERE mass_grams IS NULL").fetchone()[0]
mass_null_pct = 100.0 * n_mass_null / total_cdaw if total_cdaw else 0
check(f"mass_grams NULL rate 20-70%% (actual {mass_null_pct:.0f}%%) -- expected high", 20 < mass_null_pct < 70, warn=True)

print(f"     Total cdaw_cme_events: {total_cdaw:,}")
qf_dist = dict(con.execute("SELECT quality_flag, COUNT(*) FROM cdaw_cme_events GROUP BY quality_flag").fetchall())
print(f"     quality_flag dist: {qf_dist}")


# ===========================================================================
print()
print("=" * 65)
print("AUDIT: ingest_donki_gst -> geomagnetic_storms")
print("=" * 65)

n_total_gst = con.execute("SELECT COUNT(*) FROM geomagnetic_storms").fetchone()[0]

n_null_gst = con.execute("SELECT COUNT(*) FROM geomagnetic_storms WHERE gst_id IS NULL").fetchone()[0]
check("gst_id never NULL", n_null_gst == 0)

n_null_kp = con.execute("SELECT COUNT(*) FROM geomagnetic_storms WHERE kp_index_max IS NULL").fetchone()[0]
check("kp_index_max always populated", n_null_kp == 0)

n_kp_range = con.execute(
    "SELECT COUNT(*) FROM geomagnetic_storms WHERE kp_index_max < 1 OR kp_index_max > 9"
).fetchone()[0]
check("kp_index_max in range 1-9", n_kp_range == 0, warn=True)

n_bad_src_gst = con.execute("SELECT COUNT(*) FROM geomagnetic_storms WHERE source_catalog != 'DONKI'").fetchone()[0]
check("source_catalog always 'DONKI'", n_bad_src_gst == 0)

print(f"     Total geomagnetic_storms: {n_total_gst}")
kp_dist = con.execute("SELECT MIN(kp_index_max), MAX(kp_index_max), AVG(kp_index_max) FROM geomagnetic_storms").fetchone()
print(f"     kp_index_max: min={kp_dist[0]}, max={kp_dist[1]}, avg={kp_dist[2]:.1f}")


# ===========================================================================
print()
print("=" * 65)
print("AUDIT: ingest_donki_ips -> interplanetary_shocks")
print("=" * 65)

n_total_ips = con.execute("SELECT COUNT(*) FROM interplanetary_shocks").fetchone()[0]

n_null_ips = con.execute("SELECT COUNT(*) FROM interplanetary_shocks WHERE ips_id IS NULL").fetchone()[0]
check("ips_id never NULL", n_null_ips == 0)

n_non_earth = con.execute("SELECT COUNT(*) FROM interplanetary_shocks WHERE location != 'Earth'").fetchone()[0]
check("location always 'Earth' (API-level filter)", n_non_earth == 0)

n_bad_et = con.execute(
    "SELECT COUNT(*) FROM interplanetary_shocks WHERE event_time IS NOT NULL AND event_time NOT LIKE '%T%'"
).fetchone()[0]
check("event_time is ISO format (contains T)", n_bad_et == 0)

n_bad_src_ips = con.execute("SELECT COUNT(*) FROM interplanetary_shocks WHERE source_catalog != 'DONKI'").fetchone()[0]
check("source_catalog always 'DONKI'", n_bad_src_ips == 0)

print(f"     Total interplanetary_shocks: {n_total_ips}")


# ===========================================================================
print()
print("=" * 65)
print("AUDIT: ingest_donki_enlil -> enlil_simulations")
print("=" * 65)

tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
if "enlil_simulations" in tables:
    n_enlil = con.execute("SELECT COUNT(*) FROM enlil_simulations").fetchone()[0]
    if n_enlil > 0:
        n_null_sim = con.execute("SELECT COUNT(*) FROM enlil_simulations WHERE simulation_id IS NULL").fetchone()[0]
        check("simulation_id never NULL", n_null_sim == 0)
        n_bad_src_enlil = con.execute("SELECT COUNT(*) FROM enlil_simulations WHERE source_catalog != 'DONKI'").fetchone()[0]
        check("source_catalog always 'DONKI'", n_bad_src_enlil == 0)
    else:
        print("  WARN  enlil_simulations has 0 rows -- not ported; populated by live fetch only")
    print(f"     Total enlil_simulations: {n_enlil}")
else:
    print("  FAIL  enlil_simulations table NOT FOUND")
    issues.append(("enlil_simulations table missing", False))


# ===========================================================================
print()
print("=" * 65)
print("AUDIT: ingest_flares (GOES + DONKI) + dedup_flares")
print("=" * 65)

n_total_flares = con.execute("SELECT COUNT(*) FROM flares").fetchone()[0]

n_null_fid = con.execute("SELECT COUNT(*) FROM flares WHERE flare_id IS NULL").fetchone()[0]
check("flare_id never NULL", n_null_fid == 0)

n_ar_zero_f = con.execute("SELECT COUNT(*) FROM flares WHERE active_region_num = 0").fetchone()[0]
check("active_region_num: 0 never stored (-> None) [RULE-003]", n_ar_zero_f == 0)

bad_letter = con.execute(
    "SELECT COUNT(*) FROM flares WHERE class_letter IS NOT NULL AND class_letter NOT IN ('A','B','C','M','X')"
).fetchone()[0]
check("class_letter only A/B/C/M/X when non-null", bad_letter == 0)

n_bad_mag = con.execute(
    "SELECT COUNT(*) FROM flares WHERE class_magnitude IS NOT NULL AND class_magnitude <= 0"
).fetchone()[0]
check("class_magnitude > 0 when non-null", n_bad_mag == 0)

n_bad_bt = con.execute(
    "SELECT COUNT(*) FROM flares WHERE begin_time IS NOT NULL AND begin_time NOT LIKE '%T%' AND begin_time NOT LIKE '%-%'"
).fetchone()[0]
check("begin_time is date-like when non-null", n_bad_bt == 0)

bad_src_f = con.execute("SELECT COUNT(*) FROM flares WHERE source_catalog NOT IN ('DONKI', 'GOES')").fetchone()[0]
check("source_catalog only 'DONKI' or 'GOES'", bad_src_f == 0)

lev_rows = con.execute("SELECT flare_id, linked_event_ids FROM flares WHERE linked_event_ids IS NOT NULL LIMIT 200").fetchall()
bad_lev = sum(1 for _, lev in lev_rows if not _json_ok(lev))
check("linked_event_ids valid JSON when non-null", bad_lev == 0)

goes_count = con.execute("SELECT COUNT(*) FROM flares WHERE source_catalog = 'GOES'").fetchone()[0]
check("No GOES records present (ported data is DONKI-only; dedup not yet needed)", goes_count == 0)

gsat = con.execute("SELECT COUNT(*) FROM flares WHERE goes_satellite IS NOT NULL").fetchone()[0]
check("goes_satellite NULL (no GOES data ingested yet)", gsat == 0)

print(f"     Total flares: {n_total_flares}")
cat_dist = dict(con.execute("SELECT source_catalog, COUNT(*) FROM flares GROUP BY source_catalog").fetchall())
print(f"     source_catalog: {cat_dist}")
class_dist = dict(con.execute("SELECT class_letter, COUNT(*) FROM flares GROUP BY class_letter ORDER BY class_letter").fetchall())
print(f"     class_letter:   {class_dist}")


# ===========================================================================
print()
print("=" * 65)
print("AUDIT: solar_wind_hourly (OMNI) -- RULE-031, RULE-120")
print("=" * 65)

n_total_sw = con.execute("SELECT COUNT(*) FROM solar_wind_hourly").fetchone()[0]

n_sentinel_bz = con.execute("SELECT COUNT(*) FROM solar_wind_hourly WHERE bz_gsm > 9990").fetchone()[0]
check("bz_gsm: no sentinel leak > 9990 [RULE-120]", n_sentinel_bz == 0)

n_sentinel_flow = con.execute("SELECT COUNT(*) FROM solar_wind_hourly WHERE flow_speed > 9990").fetchone()[0]
check("flow_speed: no sentinel leak > 9990", n_sentinel_flow == 0)

n_sentinel_density = con.execute("SELECT COUNT(*) FROM solar_wind_hourly WHERE proton_density > 9990").fetchone()[0]
check("proton_density: no sentinel leak > 9990", n_sentinel_density == 0)

n_bz_present = con.execute("SELECT COUNT(*) FROM solar_wind_hourly WHERE bz_gsm IS NOT NULL").fetchone()[0]
check("bz_gsm column populated [RULE-031]", n_bz_present > 100000)

n_bz_range = con.execute(
    "SELECT COUNT(*) FROM solar_wind_hourly WHERE bz_gsm IS NOT NULL AND (bz_gsm < -500 OR bz_gsm > 500)"
).fetchone()[0]
check("bz_gsm in plausible range -500 to +500 nT", n_bz_range == 0, warn=True)

n_bad_dt_sw = con.execute("SELECT COUNT(*) FROM solar_wind_hourly WHERE datetime NOT LIKE '%-%'").fetchone()[0]
check("datetime column is date-like", n_bad_dt_sw == 0)

print(f"     Total solar_wind_hourly: {n_total_sw:,}")
bz_stats = con.execute("SELECT MIN(bz_gsm), MAX(bz_gsm), AVG(bz_gsm) FROM solar_wind_hourly WHERE bz_gsm IS NOT NULL").fetchone()
print(f"     bz_gsm: min={bz_stats[0]:.1f}, max={bz_stats[1]:.1f}, avg={bz_stats[2]:.2f} nT")


# ===========================================================================
print()
print("=" * 65)
print("AUDIT: symh_hourly")
print("=" * 65)

n_total_sym = con.execute("SELECT COUNT(*) FROM symh_hourly").fetchone()[0]

n_null_dt_sym = con.execute("SELECT COUNT(*) FROM symh_hourly WHERE datetime IS NULL").fetchone()[0]
check("datetime never NULL (PK)", n_null_dt_sym == 0)

n_sym_sentinel = con.execute("SELECT COUNT(*) FROM symh_hourly WHERE symh_nt > 9990").fetchone()[0]
check("symh_nt: no sentinel leak > 9990", n_sym_sentinel == 0)

n_sym_range = con.execute(
    "SELECT COUNT(*) FROM symh_hourly WHERE symh_nt IS NOT NULL AND (symh_nt < -600 OR symh_nt > 200)"
).fetchone()[0]
check("symh_nt in plausible range -600 to +200 nT", n_sym_range == 0, warn=True)

print(f"     Total symh_hourly: {n_total_sym:,}")
sym_stats = con.execute("SELECT MIN(symh_nt), MAX(symh_nt) FROM symh_hourly WHERE symh_nt IS NOT NULL").fetchone()
print(f"     symh_nt: min={sym_stats[0]}, max={sym_stats[1]} nT")


# ===========================================================================
print()
print("=" * 65)
print("AUDIT: kp_3hr (GFZ) -- post kp_3hr datetime fix")
print("=" * 65)

n_total_kp = con.execute("SELECT COUNT(*) FROM kp_3hr").fetchone()[0]

n_bad_dt_kp = con.execute("SELECT COUNT(*) FROM kp_3hr WHERE datetime NOT LIKE '____-__-__'").fetchone()[0]
check("datetime is YYYY-MM-DD format (post-fix)", n_bad_dt_kp == 0)

n_old_fmt = con.execute("SELECT COUNT(*) FROM kp_3hr WHERE datetime LIKE '% %'").fetchone()[0]
check("No old 'date + days_since_1932' broken format", n_old_fmt == 0)

n_kp_sentinel = con.execute("SELECT COUNT(*) FROM kp_3hr WHERE kp > 90").fetchone()[0]
check("kp: no sentinel values > 90", n_kp_sentinel == 0)

n_kp_range = con.execute("SELECT COUNT(*) FROM kp_3hr WHERE kp IS NOT NULL AND (kp < 0 OR kp > 9)").fetchone()[0]
check("kp in range 0-9 (direct scale)", n_kp_range == 0, warn=True)

n_bad_src_kp = con.execute("SELECT COUNT(*) FROM kp_3hr WHERE source_catalog != 'GFZ'").fetchone()[0]
check("source_catalog always 'GFZ'", n_bad_src_kp == 0)

print(f"     Total kp_3hr: {n_total_kp:,}")
kp_stats = con.execute("SELECT MIN(kp), MAX(kp), AVG(kp) FROM kp_3hr WHERE kp IS NOT NULL").fetchone()
print(f"     kp: min={kp_stats[0]}, max={kp_stats[1]}, avg={kp_stats[2]:.2f}")


# ===========================================================================
print()
print("=" * 65)
print("AUDIT: silso_daily_ssn")
print("=" * 65)

n_total_ssn = con.execute("SELECT COUNT(*) FROM silso_daily_ssn").fetchone()[0]

n_null_date_ssn = con.execute("SELECT COUNT(*) FROM silso_daily_ssn WHERE date IS NULL").fetchone()[0]
check("date never NULL (PK)", n_null_date_ssn == 0)

n_ssn_range = con.execute(
    "SELECT COUNT(*) FROM silso_daily_ssn WHERE sunspot_number IS NOT NULL AND (sunspot_number < 0 OR sunspot_number > 500)"
).fetchone()[0]
check("sunspot_number in range 0-500 when non-null", n_ssn_range == 0, warn=True)

n_prov_bad = con.execute(
    "SELECT COUNT(*) FROM silso_daily_ssn WHERE provisional NOT IN (0, 1) AND provisional IS NOT NULL"
).fetchone()[0]
check("provisional only 0 or 1", n_prov_bad == 0)

n_bad_src_ssn = con.execute("SELECT COUNT(*) FROM silso_daily_ssn WHERE source_catalog != 'SILSO'").fetchone()[0]
check("source_catalog always 'SILSO'", n_bad_src_ssn == 0)

print(f"     Total silso_daily_ssn: {n_total_ssn:,}")
ssn_stats = con.execute("SELECT MIN(sunspot_number), MAX(sunspot_number) FROM silso_daily_ssn WHERE sunspot_number IS NOT NULL").fetchone()
print(f"     sunspot_number: min={ssn_stats[0]}, max={ssn_stats[1]}")


# ===========================================================================
print()
print("=" * 65)
print("SUMMARY")
print("=" * 65)
hard_fails = [l for l, w in issues if not w]
warnings_list = [l for l, w in issues if w]
print(f"\nHard failures : {len(hard_fails)}")
for f in hard_fails:
    print(f"  FAIL  {f}")
print(f"Warnings      : {len(warnings_list)}")
for w in warnings_list:
    print(f"  WARN  {w}")
if not hard_fails and not warnings_list:
    print("  All checks passed. Data fully compliant with ingest script contracts.")
elif not hard_fails:
    print("  No hard failures. Warnings are expected/acceptable.")

con.close()
