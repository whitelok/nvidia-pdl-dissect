"""Summarize raw pdl_probe CSV files; use only Python standard library."""
import collections
import csv
import statistics
import sys

writer = csv.writer(sys.stdout)
writer.writerow(['file', 'pdl', 'prefetch_rows', 'early', 'n', 'e2e_median_us',
                 'e2e_sd_us', 'norm_median_us', 'overlap_median_us', 'max_abs_error'])
for filename in sys.argv[1:]:
    groups = collections.defaultdict(list)
    with open(filename) as source:
        for row in csv.DictReader(source):
            groups[tuple(row[k] for k in ('pdl', 'prefetch_rows', 'early'))].append(row)
    for key, rows in sorted(groups.items()):
        values = [float(r['e2e_us']) for r in rows]
        writer.writerow([filename, *key, len(rows), statistics.median(values),
                         statistics.stdev(values) if len(values) > 1 else 0,
                         statistics.median(float(r['norm_span_us']) for r in rows),
                         statistics.median(float(r['overlap_window_us']) for r in rows),
                         max(float(r['max_abs_error']) for r in rows)])
