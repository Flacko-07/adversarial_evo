#!/usr/bin/env python3
"""
Label OS fingerprint CSV rows using the NVD API.
Expects a CSV with at least columns: os, kernel (build number), installed_updates.
Outputs the same CSV with an added 'label' column (1 = vulnerable, 0 = not).
Usage:
  python label_os_nvd.py input.csv output.csv
"""
import csv
import os
import sys
import time
import requests

# --- Configuration ---
NVD_API_KEY = os.getenv("NVD_API_KEY", "")
NVD_BASE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
# CPE mapping for Windows 10/11 builds (example)
WINDOWS_CPE_TEMPLATE = "cpe:2.3:o:microsoft:windows_10:{version}:*:*:*:*:*:*:*"

def query_nvd(keyword, cpe=None, max_results=5):
    """Query NVD API for a keyword or CPE and return list of CVSS scores >= 7.0."""
    params = {
        "keywordSearch": keyword,
        "resultsPerPage": max_results,
    }
    if NVD_API_KEY:
        params["apiKey"] = NVD_API_KEY
    if cpe:
        params["cpeName"] = cpe

    try:
        resp = requests.get(NVD_BASE_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        vulnerabilities = data.get("vulnerabilities", [])
        critical_cves = []
        for vuln in vulnerabilities:
            metrics = vuln.get("cve", {}).get("metrics", {})
            # Check CVSS v3.1/3.0
            for m in metrics.get("cvssMetricV31", []) + metrics.get("cvssMetricV30", []):
                score = m.get("cvssData", {}).get("baseScore", 0)
                if score >= 7.0:
                    critical_cves.append(
                        f"{vuln['cve']['id']}:{score}"
                    )
        return critical_cves
    except Exception as e:
        print(f"⚠️ NVD query failed: {e}", file=sys.stderr)
        return []

def get_os_vulnerabilities(os_name, kernel_version):
    """Check for known vulnerabilities in the given OS and build."""
    if os_name.lower() == "windows":
        # Determine Windows version from build number
        build = int(kernel_version) if kernel_version.isdigit() else 0
        # Approximate mapping: 22000+ = Windows 11, 19041+ = Windows 10
        if build >= 22000:
            cpe = f"cpe:2.3:o:microsoft:windows_11:-:*:*:*:*:*:*:*"
        elif build >= 19041:
            cpe = f"cpe:2.3:o:microsoft:windows_10:-:*:*:*:*:*:*:*"
        else:
            cpe = f"cpe:2.3:o:microsoft:windows_10:*:*:*:*:*:*:*:*"
        return query_nvd("Windows", cpe=cpe)
    # Expand for Linux / macOS later
    return []

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} input.csv output.csv", file=sys.stderr)
        sys.exit(1)

    infile = sys.argv[1]
    outfile = sys.argv[2]

    with open(infile, newline='', encoding='utf-16') as fin, \
         open(outfile, 'w', newline='', encoding='utf-8') as fout:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames + ['label']
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            os_name = row.get('os', '').strip().lower()
            kernel = row.get('kernel', '').strip()
            label = 0
            # Check if any critical CVEs affect the OS
            vulns = get_os_vulnerabilities(os_name, kernel)
            if vulns:
                label = 1
                print(f"  🔴 {os_name} build {kernel} has critical CVEs: {vulns}", file=sys.stderr)
            else:
                print(f"  ✅ {os_name} build {kernel} appears up‑to‑date", file=sys.stderr)
            row['label'] = label
            writer.writerow(row)
    print("✅ Labeled OS fingerprints written to", outfile)

if __name__ == '__main__':
    main()
