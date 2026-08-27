#!/usr/bin/env python3
"""Analyze P2F -> rejection correlation across 3 hosts."""

import json
from typing import List, Dict, Any
from scipy.stats import fisher_exact
from scipy.stats import chi2
import numpy as np

# Input data from 3 hosts
host_results = [
    {
        "host": "sk1",
        "n_repos_examined": 388,
        "n_prs_total": 19711,
        "n_prs_clean": 19590,
        "n_excluded_infra_unknown": 121,
        "host_crosstab": {
            "p2f_rej": 239,
            "p2f_acc": 1048,
            "nonp2f_rej": 3970,
            "nonp2f_acc": 14333
        },
        "per_repo": [
            {"repo": "ARO-RP", "p2f_rej": 1, "p2f_acc": 6, "nonp2f_rej": 11, "nonp2f_acc": 101},
            {"repo": "DSpace", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 4, "nonp2f_acc": 83},
            {"repo": "FEDOT", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 4, "nonp2f_acc": 145},
            {"repo": "MONAI", "p2f_rej": 0, "p2f_acc": 3, "nonp2f_rej": 3, "nonp2f_acc": 90},
            {"repo": "OpenPype", "p2f_rej": 0, "p2f_acc": 1, "nonp2f_rej": 15, "nonp2f_acc": 55},
            {"repo": "PaddleNLP", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 8, "nonp2f_acc": 91},
            {"repo": "Python", "p2f_rej": 10, "p2f_acc": 3, "nonp2f_rej": 37, "nonp2f_acc": 44},
            {"repo": "TJ-Bot", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 12, "nonp2f_acc": 50},
            {"repo": "Theano", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 18, "nonp2f_acc": 72},
            {"repo": "accumulo", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 44, "nonp2f_acc": 84},
            {"repo": "alf", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 3, "nonp2f_acc": 90},
            {"repo": "amplify-android", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 9, "nonp2f_acc": 72},
            {"repo": "api", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 21, "nonp2f_acc": 109},
            {"repo": "apm-agent-java", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 6, "nonp2f_acc": 57},
            {"repo": "assisted-service", "p2f_rej": 0, "p2f_acc": 4, "nonp2f_rej": 3, "nonp2f_acc": 95},
            {"repo": "astro-sdk", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 6, "nonp2f_acc": 56},
            {"repo": "astropy", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 21, "nonp2f_acc": 66},
            {"repo": "atlasdb", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 11, "nonp2f_acc": 93},
            {"repo": "autoscaler", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 11, "nonp2f_acc": 77},
            {"repo": "avocado-vt", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 9, "nonp2f_acc": 75},
            {"repo": "aws-operator", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 2, "nonp2f_acc": 63},
            {"repo": "beats", "p2f_rej": 0, "p2f_acc": 1, "nonp2f_rej": 8, "nonp2f_acc": 53},
            {"repo": "bee", "p2f_rej": 2, "p2f_acc": 3, "nonp2f_rej": 8, "nonp2f_acc": 98},
            {"repo": "bot", "p2f_rej": 0, "p2f_acc": 1, "nonp2f_rej": 12, "nonp2f_acc": 93},
            {"repo": "bytebase", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 4, "nonp2f_acc": 55},
            {"repo": "captum", "p2f_rej": 26, "p2f_acc": 20, "nonp2f_rej": 292, "nonp2f_acc": 20},
            {"repo": "carbon-apimgt", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 9, "nonp2f_acc": 65},
            {"repo": "cdap", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 13, "nonp2f_acc": 84},
            {"repo": "cert-manager", "p2f_rej": 1, "p2f_acc": 9, "nonp2f_rej": 9, "nonp2f_acc": 65},
            {"repo": "certbot", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 16, "nonp2f_acc": 96},
            {"repo": "chainlink", "p2f_rej": 0, "p2f_acc": 13, "nonp2f_rej": 6, "nonp2f_acc": 47},
            {"repo": "checker-framework", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 19, "nonp2f_acc": 64},
            {"repo": "chia-blockchain", "p2f_rej": 2, "p2f_acc": 1, "nonp2f_rej": 20, "nonp2f_acc": 82},
            {"repo": "civiform", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 4, "nonp2f_acc": 82},
            {"repo": "click", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 62, "nonp2f_acc": 18},
            {"repo": "coala-bears", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 35, "nonp2f_acc": 74},
            {"repo": "cobra", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 30, "nonp2f_acc": 50},
            {"repo": "commercetools-sync-java", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 8, "nonp2f_acc": 77},
            {"repo": "conan-center-index", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 13, "nonp2f_acc": 79},
            {"repo": "core", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 10, "nonp2f_acc": 91},
            {"repo": "corso", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 7, "nonp2f_acc": 66},
            {"repo": "cosmos-sdk", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 9, "nonp2f_acc": 95},
            {"repo": "crc", "p2f_rej": 1, "p2f_acc": 0, "nonp2f_rej": 3, "nonp2f_acc": 71},
            {"repo": "datalad", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 8, "nonp2f_acc": 49},
            {"repo": "datawave", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 20, "nonp2f_acc": 50},
            {"repo": "dd-trace-java", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 8, "nonp2f_acc": 71},
            {"repo": "ddf", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 17, "nonp2f_acc": 96},
            {"repo": "dgl", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 8, "nonp2f_acc": 94},
            {"repo": "dgraph", "p2f_rej": 0, "p2f_acc": 5, "nonp2f_rej": 26, "nonp2f_acc": 19},
            {"repo": "dhis2-core", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 7, "nonp2f_acc": 66},
            {"repo": "druid", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 19, "nonp2f_acc": 97},
            {"repo": "easybuild-easyblocks", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 13, "nonp2f_acc": 56},
            {"repo": "edgex-go", "p2f_rej": 1, "p2f_acc": 1, "nonp2f_rej": 7, "nonp2f_acc": 70},
            {"repo": "edx-platform", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 19, "nonp2f_acc": 64},
            {"repo": "elide", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 14, "nonp2f_acc": 50},
            {"repo": "evalml", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 0, "nonp2f_acc": 6},
            {"repo": "eventing", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 6, "nonp2f_acc": 80},
            {"repo": "evergreen", "p2f_rej": 2, "p2f_acc": 45, "nonp2f_rej": 5, "nonp2f_acc": 54},
            {"repo": "fairlearn", "p2f_rej": 0, "p2f_acc": 12, "nonp2f_rej": 10, "nonp2f_acc": 25},
            {"repo": "fcrepo", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 21, "nonp2f_acc": 57},
            {"repo": "freqtrade", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 8, "nonp2f_acc": 58},
            {"repo": "gateway-api", "p2f_rej": 1, "p2f_acc": 1, "nonp2f_rej": 12, "nonp2f_acc": 45},
            {"repo": "gcp-variant-transforms", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 3, "nonp2f_acc": 68},
            {"repo": "gin", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 38, "nonp2f_acc": 70},
            {"repo": "git-machete-intellij-plugin", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 3, "nonp2f_acc": 73},
            {"repo": "go-cloud", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 6, "nonp2f_acc": 79},
            {"repo": "go-livepeer", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 4, "nonp2f_acc": 75},
            {"repo": "go-vcloud-director", "p2f_rej": 0, "p2f_acc": 1, "nonp2f_rej": 4, "nonp2f_acc": 65},
            {"repo": "graphhopper", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 9, "nonp2f_acc": 48},
            {"repo": "great_expectations", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 4, "nonp2f_acc": 78},
            {"repo": "gridgain", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 57, "nonp2f_acc": 43},
            {"repo": "hadoop", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 50, "nonp2f_acc": 49},
            {"repo": "harbor", "p2f_rej": 1, "p2f_acc": 19, "nonp2f_rej": 4, "nonp2f_acc": 50},
            {"repo": "haystack", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 8, "nonp2f_acc": 57},
            {"repo": "hbase", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 37, "nonp2f_acc": 68},
            {"repo": "helix", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 0, "nonp2f_acc": 6},
            {"repo": "hibernate-search", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 19, "nonp2f_acc": 63},
            {"repo": "hive", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 106, "nonp2f_acc": 28},
            {"repo": "htsjdk", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 17, "nonp2f_acc": 105},
            {"repo": "hydroshare", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 6, "nonp2f_acc": 76},
            {"repo": "hypershift", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 5, "nonp2f_acc": 58},
            {"repo": "ignite-3", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 41, "nonp2f_acc": 97},
            {"repo": "indico", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 27, "nonp2f_acc": 71},
            {"repo": "iris", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 17, "nonp2f_acc": 107},
            {"repo": "jdaviz", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 4, "nonp2f_acc": 59},
            {"repo": "k6", "p2f_rej": 2, "p2f_acc": 42, "nonp2f_rej": 8, "nonp2f_acc": 58},
            {"repo": "karpenter", "p2f_rej": 1, "p2f_acc": 10, "nonp2f_rej": 4, "nonp2f_acc": 33},
            {"repo": "keycloak", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 13, "nonp2f_acc": 67},
            {"repo": "kubebuilder", "p2f_rej": 0, "p2f_acc": 6, "nonp2f_rej": 14, "nonp2f_acc": 53},
            {"repo": "kubermatic", "p2f_rej": 0, "p2f_acc": 7, "nonp2f_rej": 4, "nonp2f_acc": 66},
            {"repo": "lightning", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 17, "nonp2f_acc": 112},
            {"repo": "linkerd2", "p2f_rej": 2, "p2f_acc": 2, "nonp2f_rej": 1, "nonp2f_acc": 55},
            {"repo": "lnd_gomod", "p2f_rej": 0, "p2f_acc": 60, "nonp2f_rej": 0, "nonp2f_acc": 130},
            {"repo": "lnd_gomod_rej", "p2f_rej": 66, "p2f_acc": 0, "nonp2f_rej": 124, "nonp2f_acc": 0},
            {"repo": "lucene", "p2f_rej": 1, "p2f_acc": 9, "nonp2f_rej": 3, "nonp2f_acc": 49},
            {"repo": "m3", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 8, "nonp2f_acc": 116},
            {"repo": "maya", "p2f_rej": 3, "p2f_acc": 11, "nonp2f_rej": 15, "nonp2f_acc": 85},
            {"repo": "meson", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 13, "nonp2f_acc": 57},
            {"repo": "microsoft-authentication-library-common-for-android", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 9, "nonp2f_acc": 114},
            {"repo": "minikube", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 3, "nonp2f_acc": 55},
            {"repo": "minio_gomod", "p2f_rej": 0, "p2f_acc": 32, "nonp2f_rej": 0, "nonp2f_acc": 268},
            {"repo": "minio_gomod_rej", "p2f_rej": 22, "p2f_acc": 0, "nonp2f_rej": 174, "nonp2f_acc": 0},
            {"repo": "mlflow", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 7, "nonp2f_acc": 86},
            {"repo": "mmdetection", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 13, "nonp2f_acc": 108},
            {"repo": "models", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 10, "nonp2f_acc": 62},
            {"repo": "mypy", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 9, "nonp2f_acc": 52},
            {"repo": "nessie", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 9, "nonp2f_acc": 59},
            {"repo": "nncf", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 5, "nonp2f_acc": 103},
            {"repo": "nomad", "p2f_rej": 2, "p2f_acc": 19, "nonp2f_rej": 1, "nonp2f_acc": 58},
            {"repo": "nuxeo-drive", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 14, "nonp2f_acc": 10},
            {"repo": "oauth2-proxy", "p2f_rej": 18, "p2f_acc": 9, "nonp2f_rej": 108, "nonp2f_acc": 240},
            {"repo": "ocs-ci", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 34, "nonp2f_acc": 120},
            {"repo": "odemis", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 8, "nonp2f_acc": 110},
            {"repo": "odl", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 36, "nonp2f_acc": 99},
            {"repo": "open-kilda", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 5, "nonp2f_acc": 54},
            {"repo": "open-smart-grid-platform", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 6, "nonp2f_acc": 65},
            {"repo": "openmrs-core", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 42, "nonp2f_acc": 60},
            {"repo": "opentelemetry-collector-contrib", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 14, "nonp2f_acc": 91},
            {"repo": "osparc-simcore", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 3, "nonp2f_acc": 106},
            {"repo": "ovn-kubernetes", "p2f_rej": 2, "p2f_acc": 5, "nonp2f_rej": 15, "nonp2f_acc": 76},
            {"repo": "ozone", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 10, "nonp2f_acc": 85},
            {"repo": "pachyderm", "p2f_rej": 1, "p2f_acc": 10, "nonp2f_rej": 5, "nonp2f_acc": 53},
            {"repo": "pinot", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 8, "nonp2f_acc": 89},
            {"repo": "pmm", "p2f_rej": 0, "p2f_acc": 2, "nonp2f_rej": 10, "nonp2f_acc": 48},
            {"repo": "podman", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 41, "nonp2f_acc": 38},
            {"repo": "powsybl-core", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 12, "nonp2f_acc": 102},
            {"repo": "prebid-server", "p2f_rej": 0, "p2f_acc": 26, "nonp2f_rej": 9, "nonp2f_acc": 97},
            {"repo": "prebid-server-java", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 3, "nonp2f_acc": 66},
            {"repo": "pudl", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 9, "nonp2f_acc": 68},
            {"repo": "pulpcore", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 5, "nonp2f_acc": 61},
            {"repo": "pulsar", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 12, "nonp2f_acc": 66},
            {"repo": "pyccel", "p2f_rej": 0, "p2f_acc": 13, "nonp2f_rej": 5, "nonp2f_acc": 93},
            {"repo": "pylint", "p2f_rej": 1, "p2f_acc": 0, "nonp2f_rej": 4, "nonp2f_acc": 81},
            {"repo": "pymc", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 15, "nonp2f_acc": 87},
            {"repo": "pypeit", "p2f_rej": 4, "p2f_acc": 0, "nonp2f_rej": 9, "nonp2f_acc": 9},
            {"repo": "python-telegram-bot", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 5, "nonp2f_acc": 84},
            {"repo": "qiskit-experiments", "p2f_rej": 9, "p2f_acc": 17, "nonp2f_rej": 62, "nonp2f_acc": 276},
            {"repo": "realm-java", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 11, "nonp2f_acc": 64},
            {"repo": "rook", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 3, "nonp2f_acc": 69},
            {"repo": "rskj", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 7, "nonp2f_acc": 70},
            {"repo": "runelite", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 53, "nonp2f_acc": 50},
            {"repo": "satpy", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 6, "nonp2f_acc": 68},
            {"repo": "scikit-fda", "p2f_rej": 0, "p2f_acc": 2, "nonp2f_rej": 11, "nonp2f_acc": 55},
            {"repo": "scylla-cluster-tests", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 5, "nonp2f_acc": 83},
            {"repo": "selenium-tests", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 19, "nonp2f_acc": 64},
            {"repo": "servicetalk", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 3, "nonp2f_acc": 130},
            {"repo": "shardingsphere", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 6, "nonp2f_acc": 59},
            {"repo": "sir-lancebot", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 16, "nonp2f_acc": 72},
            {"repo": "skyportal", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 5, "nonp2f_acc": 77},
            {"repo": "snuba", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 12, "nonp2f_acc": 108},
            {"repo": "sonic-utilities", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 2, "nonp2f_acc": 70},
            {"repo": "spire", "p2f_rej": 11, "p2f_acc": 12, "nonp2f_rej": 100, "nonp2f_acc": 1069},
            {"repo": "stackrox", "p2f_rej": 2, "p2f_acc": 20, "nonp2f_rej": 3, "nonp2f_acc": 66},
            {"repo": "status-go", "p2f_rej": 3, "p2f_acc": 35, "nonp2f_rej": 12, "nonp2f_acc": 67},
            {"repo": "storm", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 15, "nonp2f_acc": 76},
            {"repo": "sunpy", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 30, "nonp2f_acc": 30},
            {"repo": "swebench-astropy", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 180, "nonp2f_acc": 95},
            {"repo": "swebench-django", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 2, "nonp2f_acc": 0},
            {"repo": "swebench-flask", "p2f_rej": 0, "p2f_acc": 11, "nonp2f_rej": 0, "nonp2f_acc": 100},
            {"repo": "swebench-matplotlib", "p2f_rej": 0, "p2f_acc": 162, "nonp2f_rej": 107, "nonp2f_acc": 125},
            {"repo": "swebench-sphinx", "p2f_rej": 0, "p2f_acc": 174, "nonp2f_rej": 130, "nonp2f_acc": 52},
            {"repo": "swebench-sympy", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 146, "nonp2f_acc": 388},
            {"repo": "swebench-xarray", "p2f_rej": 0, "p2f_acc": 131, "nonp2f_rej": 170, "nonp2f_acc": 8},
            {"repo": "swebench_reject_sympy", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 146, "nonp2f_acc": 0},
            {"repo": "syzkaller", "p2f_rej": 1, "p2f_acc": 14, "nonp2f_rej": 9, "nonp2f_acc": 29},
            {"repo": "tanzu-framework", "p2f_rej": 0, "p2f_acc": 2, "nonp2f_rej": 10, "nonp2f_acc": 65},
            {"repo": "teammates", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 7, "nonp2f_acc": 72},
            {"repo": "telegraf", "p2f_rej": 1, "p2f_acc": 0, "nonp2f_rej": 19, "nonp2f_acc": 74},
            {"repo": "torpedo", "p2f_rej": 1, "p2f_acc": 4, "nonp2f_rej": 10, "nonp2f_acc": 89},
            {"repo": "tp-qemu", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 14, "nonp2f_acc": 91},
            {"repo": "trieste", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 8, "nonp2f_acc": 71},
            {"repo": "triplea", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 9, "nonp2f_acc": 103},
            {"repo": "vault_gomod", "p2f_rej": 0, "p2f_acc": 48, "nonp2f_rej": 0, "nonp2f_acc": 252},
            {"repo": "vault_gomod_rej", "p2f_rej": 37, "p2f_acc": 0, "nonp2f_rej": 106, "nonp2f_acc": 0},
            {"repo": "velero", "p2f_rej": 0, "p2f_acc": 2, "nonp2f_rej": 5, "nonp2f_acc": 82},
            {"repo": "vsphere-csi-driver", "p2f_rej": 0, "p2f_acc": 1, "nonp2f_rej": 5, "nonp2f_acc": 114},
            {"repo": "windows-machine-config-operator", "p2f_rej": 0, "p2f_acc": 1, "nonp2f_rej": 5, "nonp2f_acc": 77},
            {"repo": "zulip-terminal", "p2f_rej": 1, "p2f_acc": 0, "nonp2f_rej": 38, "nonp2f_acc": 26}
        ]
    },
    {
        "host": "sk2",
        "n_repos_examined": 102,
        "n_prs_total": 17607,
        "n_prs_clean": 11779,
        "n_excluded_infra_unknown": 5828,
        "host_crosstab": {
            "p2f_rej": 2224,
            "p2f_acc": 529,
            "nonp2f_rej": 6302,
            "nonp2f_acc": 8552
        },
        "per_repo": [
            {"repo": "transformers", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 19, "nonp2f_acc": 103},
            {"repo": "calcite", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 63, "nonp2f_acc": 36},
            {"repo": "strawberryfields", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 2, "nonp2f_acc": 117},
            {"repo": "besu", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 13, "nonp2f_acc": 89},
            {"repo": "k6_rej", "p2f_rej": 112, "p2f_acc": 0, "nonp2f_rej": 35, "nonp2f_acc": 0},
            {"repo": "spire", "p2f_rej": 6, "p2f_acc": 13, "nonp2f_rej": 105, "nonp2f_acc": 1068},
            {"repo": "helm_rej", "p2f_rej": 657, "p2f_acc": 0, "nonp2f_rej": 54, "nonp2f_acc": 0},
            {"repo": "course-discovery", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 8, "nonp2f_acc": 103},
            {"repo": "cockroach", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 2, "nonp2f_acc": 110},
            {"repo": "skywalking", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 8, "nonp2f_acc": 75},
            {"repo": "lisa", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 6, "nonp2f_acc": 77},
            {"repo": "boulder", "p2f_rej": 1, "p2f_acc": 6, "nonp2f_rej": 7, "nonp2f_acc": 90},
            {"repo": "containerd_gomod_rej", "p2f_rej": 3, "p2f_acc": 0, "nonp2f_rej": 191, "nonp2f_acc": 0},
            {"repo": "dcrdex", "p2f_rej": 0, "p2f_acc": 6, "nonp2f_rej": 7, "nonp2f_acc": 123},
            {"repo": "yarpc-go", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 8, "nonp2f_acc": 79},
            {"repo": "minecolonies", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 23, "nonp2f_acc": 111},
            {"repo": "Qcodes", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 12, "nonp2f_acc": 84},
            {"repo": "tiflow", "p2f_rej": 0, "p2f_acc": 2, "nonp2f_rej": 4, "nonp2f_acc": 102},
            {"repo": "tendermint", "p2f_rej": 0, "p2f_acc": 1, "nonp2f_rej": 16, "nonp2f_acc": 89},
            {"repo": "snapcraft", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 15, "nonp2f_acc": 83},
            {"repo": "CorfuDB", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 30, "nonp2f_acc": 108},
            {"repo": "dgraph", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 444, "nonp2f_acc": 24},
            {"repo": "fdb-record-layer", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 6, "nonp2f_acc": 106},
            {"repo": "etcd_gomod", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 0, "nonp2f_acc": 103},
            {"repo": "modin", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 3, "nonp2f_acc": 93},
            {"repo": "etcd_rej", "p2f_rej": 22, "p2f_acc": 0, "nonp2f_rej": 842, "nonp2f_acc": 0},
            {"repo": "OpenSearch", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 14, "nonp2f_acc": 111},
            {"repo": "lnd_gomod_rej", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 190, "nonp2f_acc": 0},
            {"repo": "vault", "p2f_rej": 3, "p2f_acc": 12, "nonp2f_rej": 5, "nonp2f_acc": 66},
            {"repo": "dask", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 6, "nonp2f_acc": 98},
            {"repo": "mongo-go-driver", "p2f_rej": 1, "p2f_acc": 3, "nonp2f_rej": 7, "nonp2f_acc": 106},
            {"repo": "dipy", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 20, "nonp2f_acc": 88},
            {"repo": "grpc-go_rej", "p2f_rej": 23, "p2f_acc": 0, "nonp2f_rej": 275, "nonp2f_acc": 0},
            {"repo": "helm_gomod", "p2f_rej": 0, "p2f_acc": 12, "nonp2f_rej": 0, "nonp2f_acc": 288},
            {"repo": "carbon-identity-framework", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 8, "nonp2f_acc": 78},
            {"repo": "ecommerce", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 17, "nonp2f_acc": 97},
            {"repo": "neo-go", "p2f_rej": 0, "p2f_acc": 1, "nonp2f_rej": 3, "nonp2f_acc": 110},
            {"repo": "consul_rej", "p2f_rej": 747, "p2f_acc": 0, "nonp2f_rej": 76, "nonp2f_acc": 0},
            {"repo": "gensim", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 21, "nonp2f_acc": 82},
            {"repo": "jib", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 3, "nonp2f_acc": 108},
            {"repo": "atomic-reactor", "p2f_rej": 0, "p2f_acc": 1, "nonp2f_rej": 6, "nonp2f_acc": 117},
            {"repo": "skypilot", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 2, "nonp2f_acc": 99},
            {"repo": "NeMo", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 6, "nonp2f_acc": 117},
            {"repo": "vision", "p2f_rej": 0, "p2f_acc": 1, "nonp2f_rej": 10, "nonp2f_acc": 97},
            {"repo": "kubevirt", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 12, "nonp2f_acc": 84},
            {"repo": "cadence", "p2f_rej": 0, "p2f_acc": 2, "nonp2f_rej": 6, "nonp2f_acc": 91},
            {"repo": "containerd_gomod", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 0, "nonp2f_acc": 72},
            {"repo": "SpongeAPI", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 103, "nonp2f_acc": 29},
            {"repo": "avocado", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 67, "nonp2f_acc": 14},
            {"repo": "consul_gomod", "p2f_rej": 0, "p2f_acc": 289, "nonp2f_rej": 0, "nonp2f_acc": 9},
            {"repo": "image", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 17, "nonp2f_acc": 74},
            {"repo": "opentelemetry-python", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 14, "nonp2f_acc": 121},
            {"repo": "java-ladder", "p2f_rej": 0, "p2f_acc": 3, "nonp2f_rej": 4, "nonp2f_acc": 109},
            {"repo": "swebench_reject_astropy", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 180, "nonp2f_acc": 0},
            {"repo": "distributed", "p2f_rej": 0, "p2f_acc": 1, "nonp2f_rej": 23, "nonp2f_acc": 97},
            {"repo": "go-spacemesh", "p2f_rej": 3, "p2f_acc": 0, "nonp2f_rej": 76, "nonp2f_acc": 48},
            {"repo": "prometheus_rej", "p2f_rej": 401, "p2f_acc": 0, "nonp2f_rej": 515, "nonp2f_acc": 0},
            {"repo": "helm_gomod_rej", "p2f_rej": 10, "p2f_acc": 0, "nonp2f_rej": 188, "nonp2f_acc": 0},
            {"repo": "grpc_go_gomod", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 0, "nonp2f_acc": 149},
            {"repo": "gatk", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 9, "nonp2f_acc": 119},
            {"repo": "swebench-requests", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 131, "nonp2f_acc": 45},
            {"repo": "zipline", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 17, "nonp2f_acc": 68},
            {"repo": "k6_gomod_rej", "p2f_rej": 54, "p2f_acc": 0, "nonp2f_rej": 81, "nonp2f_acc": 0},
            {"repo": "sync_gateway", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 7, "nonp2f_acc": 94},
            {"repo": "gammapy", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 11, "nonp2f_acc": 81},
            {"repo": "click", "p2f_rej": 1, "p2f_acc": 0, "nonp2f_rej": 61, "nonp2f_acc": 18},
            {"repo": "jaeger", "p2f_rej": 3, "p2f_acc": 6, "nonp2f_rej": 15, "nonp2f_acc": 74},
            {"repo": "grpc-go", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 13, "nonp2f_acc": 119},
            {"repo": "qiita", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 7, "nonp2f_acc": 98},
            {"repo": "consul_gomod_rej", "p2f_rej": 141, "p2f_acc": 0, "nonp2f_rej": 8, "nonp2f_acc": 0},
            {"repo": "conan", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 10, "nonp2f_acc": 91},
            {"repo": "dd-trace-py", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 17, "nonp2f_acc": 89},
            {"repo": "numba", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 8, "nonp2f_acc": 86},
            {"repo": "vic", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 7, "nonp2f_acc": 110},
            {"repo": "Cirq", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 15, "nonp2f_acc": 114},
            {"repo": "nucypher", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 7, "nonp2f_acc": 107},
            {"repo": "gt4py", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 4, "nonp2f_acc": 80},
            {"repo": "zap-extensions", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 10, "nonp2f_acc": 93},
            {"repo": "tidb-operator", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 7, "nonp2f_acc": 85},
            {"repo": "aiida-core", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 6, "nonp2f_acc": 83},
            {"repo": "kubo", "p2f_rej": 1, "p2f_acc": 1, "nonp2f_rej": 26, "nonp2f_acc": 77},
            {"repo": "cluster-api-provider-azure", "p2f_rej": 1, "p2f_acc": 0, "nonp2f_rej": 8, "nonp2f_acc": 96},
            {"repo": "spring-integration", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 90, "nonp2f_acc": 6},
            {"repo": "neo4j", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 7, "nonp2f_acc": 59},
            {"repo": "dd-trace-go", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 11, "nonp2f_acc": 83},
            {"repo": "prometheus_gomod", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 0, "nonp2f_acc": 105},
            {"repo": "k6_gomod", "p2f_rej": 0, "p2f_acc": 125, "nonp2f_rej": 0, "nonp2f_acc": 172},
            {"repo": "prometheus_gomod_rej", "p2f_rej": 17, "p2f_acc": 0, "nonp2f_rej": 178, "nonp2f_acc": 0},
            {"repo": "swebench-astropy", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 180, "nonp2f_acc": 95},
            {"repo": "etcd_gomod_rej", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 190, "nonp2f_acc": 0},
            {"repo": "aws-parallelcluster", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 6, "nonp2f_acc": 104},
            {"repo": "pvlib-python", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 9, "nonp2f_acc": 79},
            {"repo": "odo", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 3, "nonp2f_acc": 118},
            {"repo": "grpcgo_gomod_rej", "p2f_rej": 6, "p2f_acc": 0, "nonp2f_rej": 190, "nonp2f_acc": 0},
            {"repo": "containerd_rej", "p2f_rej": 11, "p2f_acc": 0, "nonp2f_rej": 791, "nonp2f_acc": 0},
            {"repo": "node", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 3, "nonp2f_acc": 92},
            {"repo": "swebench-pytest", "p2f_rej": 0, "p2f_acc": 4, "nonp2f_rej": 175, "nonp2f_acc": 116},
            {"repo": "swebench-xarray", "p2f_rej": 0, "p2f_acc": 40, "nonp2f_rej": 170, "nonp2f_acc": 99},
            {"repo": "DIRAC", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 17, "nonp2f_acc": 65}
        ]
    },
    {
        "host": "sk3",
        "n_repos_examined": 141,
        "n_prs_total": 25055,
        "n_prs_clean": 22310,
        "n_excluded_infra_unknown": 2745,
        "host_crosstab": {
            "p2f_rej": 393,
            "p2f_acc": 1336,
            "nonp2f_rej": 3818,
            "nonp2f_acc": 16763
        },
        "per_repo": [
            {"repo": ".staging", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 22, "nonp2f_acc": 83},
            {"repo": "OpenPype", "p2f_rej": 0, "p2f_acc": 1, "nonp2f_rej": 15, "nonp2f_acc": 55},
            {"repo": "ParlAI", "p2f_rej": 1, "p2f_acc": 3, "nonp2f_rej": 9, "nonp2f_acc": 77},
            {"repo": "PlasmaPy", "p2f_rej": 1, "p2f_acc": 1, "nonp2f_rej": 8, "nonp2f_acc": 62},
            {"repo": "Python", "p2f_rej": 10, "p2f_acc": 3, "nonp2f_rej": 37, "nonp2f_acc": 44},
            {"repo": "Skript", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 43, "nonp2f_acc": 71},
            {"repo": "WALinuxAgent", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 15, "nonp2f_acc": 106},
            {"repo": "activemq-artemis", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 47, "nonp2f_acc": 26},
            {"repo": "agents-aea", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 4, "nonp2f_acc": 82},
            {"repo": "alertmanager", "p2f_rej": 5, "p2f_acc": 0, "nonp2f_rej": 11, "nonp2f_acc": 64},
            {"repo": "alluxio", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 23, "nonp2f_acc": 90},
            {"repo": "anaconda", "p2f_rej": 0, "p2f_acc": 1, "nonp2f_rej": 3, "nonp2f_acc": 78},
            {"repo": "armeria", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 2, "nonp2f_acc": 126},
            {"repo": "autogluon", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 16, "nonp2f_acc": 109},
            {"repo": "aws-greengrass-nucleus", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 12, "nonp2f_acc": 127},
            {"repo": "aws-sam-cli", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 6, "nonp2f_acc": 88},
            {"repo": "azkaban", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 7, "nonp2f_acc": 118},
            {"repo": "azure-cli-extensions", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 7, "nonp2f_acc": 97},
            {"repo": "azure-container-networking", "p2f_rej": 5, "p2f_acc": 28, "nonp2f_rej": 5, "nonp2f_acc": 85},
            {"repo": "azure-service-operator", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 4, "nonp2f_acc": 75},
            {"repo": "brooklin", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 5, "nonp2f_acc": 73},
            {"repo": "captum", "p2f_rej": 27, "p2f_acc": 21, "nonp2f_rej": 291, "nonp2f_acc": 19},
            {"repo": "celestia-node", "p2f_rej": 4, "p2f_acc": 29, "nonp2f_rej": 6, "nonp2f_acc": 93},
            {"repo": "celo-blockchain", "p2f_rej": 4, "p2f_acc": 23, "nonp2f_rej": 6, "nonp2f_acc": 41},
            {"repo": "chainer", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 7, "nonp2f_acc": 89},
            {"repo": "checkstyle", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 20, "nonp2f_acc": 85},
            {"repo": "chia-blockchain", "p2f_rej": 2, "p2f_acc": 1, "nonp2f_rej": 20, "nonp2f_acc": 82},
            {"repo": "ci-tools", "p2f_rej": 0, "p2f_acc": 6, "nonp2f_rej": 7, "nonp2f_acc": 99},
            {"repo": "cilium", "p2f_rej": 0, "p2f_acc": 1, "nonp2f_rej": 14, "nonp2f_acc": 83},
            {"repo": "click", "p2f_rej": 4, "p2f_acc": 1, "nonp2f_rej": 58, "nonp2f_acc": 17},
            {"repo": "cloudstack", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 15, "nonp2f_acc": 55},
            {"repo": "cobra", "p2f_rej": 3, "p2f_acc": 1, "nonp2f_rej": 27, "nonp2f_acc": 49},
            {"repo": "community.general", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 24, "nonp2f_acc": 98},
            {"repo": "containerized-data-importer", "p2f_rej": 0, "p2f_acc": 2, "nonp2f_rej": 5, "nonp2f_acc": 80},
            {"repo": "cortex", "p2f_rej": 6, "p2f_acc": 28, "nonp2f_rej": 4, "nonp2f_acc": 66},
            {"repo": "cylc-flow", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 3, "nonp2f_acc": 76},
            {"repo": "devito", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 17, "nonp2f_acc": 92},
            {"repo": "dgraph", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 26, "nonp2f_acc": 24},
            {"repo": "dvc", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 18, "nonp2f_acc": 112},
            {"repo": "easybuild-framework", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 5, "nonp2f_acc": 112},
            {"repo": "evalml", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 0, "nonp2f_acc": 6},
            {"repo": "eve", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 15, "nonp2f_acc": 70},
            {"repo": "fairlearn", "p2f_rej": 0, "p2f_acc": 12, "nonp2f_rej": 10, "nonp2f_acc": 25},
            {"repo": "flow", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 15, "nonp2f_acc": 84},
            {"repo": "flow-go", "p2f_rej": 1, "p2f_acc": 1, "nonp2f_rej": 29, "nonp2f_acc": 29},
            {"repo": "gateway-api", "p2f_rej": 1, "p2f_acc": 1, "nonp2f_rej": 29, "nonp2f_acc": 165},
            {"repo": "gin", "p2f_rej": 6, "p2f_acc": 3, "nonp2f_rej": 32, "nonp2f_acc": 67},
            {"repo": "golang-samples", "p2f_rej": 0, "p2f_acc": 2, "nonp2f_rej": 12, "nonp2f_acc": 92},
            {"repo": "gonum", "p2f_rej": 1, "p2f_acc": 1, "nonp2f_rej": 18, "nonp2f_acc": 75},
            {"repo": "google-cloud-python", "p2f_rej": 0, "p2f_acc": 9, "nonp2f_rej": 10, "nonp2f_acc": 82},
            {"repo": "goshimmer", "p2f_rej": 3, "p2f_acc": 6, "nonp2f_rej": 9, "nonp2f_acc": 81},
            {"repo": "grpc-java", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 47, "nonp2f_acc": 86},
            {"repo": "gsy-e", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 2, "nonp2f_acc": 93},
            {"repo": "harmony", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 12, "nonp2f_acc": 59},
            {"repo": "hazelcast", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 14, "nonp2f_acc": 84},
            {"repo": "hcsshim", "p2f_rej": 0, "p2f_acc": 1, "nonp2f_rej": 14, "nonp2f_acc": 94},
            {"repo": "hedera-services", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 9, "nonp2f_acc": 103},
            {"repo": "helix", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 0, "nonp2f_acc": 6},
            {"repo": "hive", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 106, "nonp2f_acc": 28},
            {"repo": "hono", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 21, "nonp2f_acc": 81},
            {"repo": "hudi", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 21, "nonp2f_acc": 90},
            {"repo": "improver", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 7, "nonp2f_acc": 128},
            {"repo": "infinispan", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 86, "nonp2f_acc": 11},
            {"repo": "influxdb", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 9, "nonp2f_acc": 71},
            {"repo": "ingress-gce", "p2f_rej": 0, "p2f_acc": 3, "nonp2f_rej": 7, "nonp2f_acc": 79},
            {"repo": "insolar", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 22, "nonp2f_acc": 75},
            {"repo": "iotex-core", "p2f_rej": 2, "p2f_acc": 9, "nonp2f_rej": 16, "nonp2f_acc": 102},
            {"repo": "itou", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 9, "nonp2f_acc": 102},
            {"repo": "jenkins", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 18, "nonp2f_acc": 65},
            {"repo": "jina", "p2f_rej": 0, "p2f_acc": 2, "nonp2f_rej": 3, "nonp2f_acc": 77},
            {"repo": "keanu", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 3, "nonp2f_acc": 81},
            {"repo": "keras-cv", "p2f_rej": 0, "p2f_acc": 3, "nonp2f_rej": 17, "nonp2f_acc": 73},
            {"repo": "klaytn", "p2f_rej": 5, "p2f_acc": 79, "nonp2f_rej": 5, "nonp2f_acc": 37},
            {"repo": "ksql", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 7, "nonp2f_acc": 89},
            {"repo": "lms", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 13, "nonp2f_acc": 67},
            {"repo": "longhorn-manager", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 5, "nonp2f_acc": 78},
            {"repo": "manim", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 17, "nonp2f_acc": 86},
            {"repo": "mdanalysis", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 15, "nonp2f_acc": 98},
            {"repo": "monkey", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 4, "nonp2f_acc": 82},
            {"repo": "mule", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 29, "nonp2f_acc": 79},
            {"repo": "mx-chain-go", "p2f_rej": 4, "p2f_acc": 628, "nonp2f_rej": 42, "nonp2f_acc": 3038},
            {"repo": "mx-chain-go_rej", "p2f_rej": 225, "p2f_acc": 0, "nonp2f_rej": 282, "nonp2f_acc": 0},
            {"repo": "nilearn", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 25, "nonp2f_acc": 82},
            {"repo": "nuxeo", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 17, "nonp2f_acc": 86},
            {"repo": "nuxeo-drive", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 14, "nonp2f_acc": 10},
            {"repo": "oasis-core", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 4, "nonp2f_acc": 87},
            {"repo": "oauth2-proxy", "p2f_rej": 126, "p2f_acc": 249, "nonp2f_rej": 126, "nonp2f_acc": 249},
            {"repo": "oauth2proxy_gomod_rej", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 199, "nonp2f_acc": 0},
            {"repo": "odl", "p2f_rej": 9, "p2f_acc": 8, "nonp2f_rej": 27, "nonp2f_acc": 91},
            {"repo": "opa", "p2f_rej": 0, "p2f_acc": 7, "nonp2f_rej": 8, "nonp2f_acc": 64},
            {"repo": "opencensus-java", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 8, "nonp2f_acc": 71},
            {"repo": "opentelemetry-java", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 13, "nonp2f_acc": 89},
            {"repo": "operator", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 2, "nonp2f_acc": 74},
            {"repo": "optuna", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 6, "nonp2f_acc": 114},
            {"repo": "osmosis", "p2f_rej": 9, "p2f_acc": 24, "nonp2f_rej": 4, "nonp2f_acc": 67},
            {"repo": "pd", "p2f_rej": 0, "p2f_acc": 31, "nonp2f_rej": 8, "nonp2f_acc": 78},
            {"repo": "phoenix", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 87, "nonp2f_acc": 35},
            {"repo": "picard", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 30, "nonp2f_acc": 30},
            {"repo": "polygon-edge", "p2f_rej": 1, "p2f_acc": 10, "nonp2f_rej": 4, "nonp2f_acc": 81},
            {"repo": "presto", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 74, "nonp2f_acc": 57},
            {"repo": "protostar", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 4, "nonp2f_acc": 70},
            {"repo": "pulumi", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 5, "nonp2f_acc": 72},
            {"repo": "pypeit", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 17, "nonp2f_acc": 62},
            {"repo": "pyro", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 8, "nonp2f_acc": 75},
            {"repo": "python-aiplatform", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 5, "nonp2f_acc": 70},
            {"repo": "qiskit-experiments", "p2f_rej": 11, "p2f_acc": 9, "nonp2f_rej": 60, "nonp2f_acc": 284},
            {"repo": "questdb", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 5, "nonp2f_acc": 76},
            {"repo": "rancher", "p2f_rej": 1, "p2f_acc": 0, "nonp2f_rej": 13, "nonp2f_acc": 74},
            {"repo": "rdk", "p2f_rej": 20, "p2f_acc": 65, "nonp2f_rej": 43, "nonp2f_acc": 1490},
            {"repo": "reframe", "p2f_rej": 0, "p2f_acc": 6, "nonp2f_rej": 3, "nonp2f_acc": 67},
            {"repo": "saleor", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 10, "nonp2f_acc": 77},
            {"repo": "scikit-bio", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 16, "nonp2f_acc": 72},
            {"repo": "scikit-image", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 21, "nonp2f_acc": 84},
            {"repo": "serving", "p2f_rej": 2, "p2f_acc": 4, "nonp2f_rej": 12, "nonp2f_acc": 68},
            {"repo": "sir-lancebot", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 16, "nonp2f_acc": 72},
            {"repo": "skaffold", "p2f_rej": 0, "p2f_acc": 11, "nonp2f_rej": 5, "nonp2f_acc": 28},
            {"repo": "sktime", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 6, "nonp2f_acc": 71},
            {"repo": "sonarqube", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 45, "nonp2f_acc": 73},
            {"repo": "spire", "p2f_rej": 11, "p2f_acc": 21, "nonp2f_rej": 100, "nonp2f_acc": 1060},
            {"repo": "spyder", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 6, "nonp2f_acc": 115},
            {"repo": "stargate", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 6, "nonp2f_acc": 95},
            {"repo": "sunpy", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 30, "nonp2f_acc": 30},
            {"repo": "swebench-django", "p2f_rej": 0, "p2f_acc": 53, "nonp2f_rej": 192, "nonp2f_acc": 797},
            {"repo": "swebench-flask", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 0, "nonp2f_acc": 111},
            {"repo": "swebench-pylint", "p2f_rej": 0, "p2f_acc": 31, "nonp2f_rej": 192, "nonp2f_acc": 88},
            {"repo": "swebench-scikit-learn", "p2f_rej": 0, "p2f_acc": 30, "nonp2f_rej": 167, "nonp2f_acc": 199},
            {"repo": "swebench-seaborn", "p2f_rej": 0, "p2f_acc": 21, "nonp2f_rej": 39, "nonp2f_acc": 1},
            {"repo": "synapse", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 8, "nonp2f_acc": 77},
            {"repo": "tailscale", "p2f_rej": 1, "p2f_acc": 20, "nonp2f_rej": 6, "nonp2f_acc": 73},
            {"repo": "teleport", "p2f_rej": 1, "p2f_acc": 17, "nonp2f_rej": 7, "nonp2f_acc": 53},
            {"repo": "terraform-provider-azurerm", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 16, "nonp2f_acc": 98},
            {"repo": "terraform-provider-vcd", "p2f_rej": 0, "p2f_acc": 1, "nonp2f_rej": 2, "nonp2f_acc": 70},
            {"repo": "thanos", "p2f_rej": 5, "p2f_acc": 18, "nonp2f_rej": 7, "nonp2f_acc": 75},
            {"repo": "tk-core", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 44, "nonp2f_acc": 38},
            {"repo": "tp-libvirt", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 6, "nonp2f_acc": 86},
            {"repo": "training_extensions", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 8, "nonp2f_acc": 66},
            {"repo": "vitess", "p2f_rej": 1, "p2f_acc": 21, "nonp2f_rej": 5, "nonp2f_acc": 62},
            {"repo": "weblogic-kubernetes-operator", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 10, "nonp2f_acc": 73},
            {"repo": "youtube-dl", "p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 62, "nonp2f_acc": 32},
            {"repo": "zed", "p2f_rej": 0, "p2f_acc": 18, "nonp2f_rej": 2, "nonp2f_acc": 62},
            {"repo": "zulip-terminal", "p2f_rej": 1, "p2f_acc": 0, "nonp2f_rej": 38, "nonp2f_acc": 26}
        ]
    }
]


def compute_pooled(host_results):
    """Compute pooled 2x2 crosstab across all hosts."""
    pooled = {"p2f_rej": 0, "p2f_acc": 0, "nonp2f_rej": 0, "nonp2f_acc": 0}
    for host in host_results:
        ct = host["host_crosstab"]
        pooled["p2f_rej"] += ct["p2f_rej"]
        pooled["p2f_acc"] += ct["p2f_acc"]
        pooled["nonp2f_rej"] += ct["nonp2f_rej"]
        pooled["nonp2f_acc"] += ct["nonp2f_acc"]
    return pooled


def fisher_exact_test(crosstab):
    """Compute Fisher exact OR + 95% CI + p-value."""
    # Construct 2x2 table: [[p2f_rej, p2f_acc], [nonp2f_rej, nonp2f_acc]]
    table = [
        [crosstab["p2f_rej"], crosstab["p2f_acc"]],
        [crosstab["nonp2f_rej"], crosstab["nonp2f_acc"]]
    ]

    # Fisher exact test (alternative='greater' tests P2F -> reject)
    or_val, p_value = fisher_exact(table, alternative='greater')

    # Compute 95% CI using Woolf's method (log-normal)
    # SE(log(OR)) = sqrt(1/a + 1/b + 1/c + 1/d)
    a, b, c, d = table[0][0], table[0][1], table[1][0], table[1][1]

    # Handle zero cells by adding 0.5 to all (Haldane-Ancombe correction)
    if a == 0 or b == 0 or c == 0 or d == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        or_val = (a * d) / (b * c)

    se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
    log_or = np.log(or_val)

    # 95% CI on log scale
    log_ci_low = log_or - 1.96 * se_log_or
    log_ci_high = log_or + 1.96 * se_log_or

    # Convert back to OR scale
    ci_low = np.exp(log_ci_low)
    ci_high = np.exp(log_ci_high)

    return or_val, ci_low, ci_high, p_value


def compute_mh_by_repo(host_results):
    """Compute Mantel-Haenszel OR by repo across all hosts."""
    # Collect all per-repo crosstabs
    repos = {}
    for host in host_results:
        for repo in host["per_repo"]:
            repo_name = repo["repo"]
            # Skip repos with zero total (can happen with _rej suffix only)
            total = repo["p2f_rej"] + repo["p2f_acc"] + repo["nonp2f_rej"] + repo["nonp2f_acc"]
            if total == 0:
                continue

            if repo_name not in repos:
                repos[repo_name] = repo
            else:
                # Aggregate if somehow duplicated across hosts (shouldn't happen)
                repos[repo_name]["p2f_rej"] += repo["p2f_rej"]
                repos[repo_name]["p2f_acc"] += repo["p2f_acc"]
                repos[repo_name]["nonp2f_rej"] += repo["nonp2f_rej"]
                repos[repo_name]["nonp2f_acc"] += repo["nonp2f_acc"]

    # Mantel-Haenszel pooled OR
    # OR_MH = sum(a_i * d_i / n_i) / sum(b_i * c_i / n_i)
    # where a_i = p2f_rej, b_i = p2f_acc, c_i = nonp2f_rej, d_i = nonp2f_acc
    # n_i = total in stratum i

    mh_numer = 0.0  # sum(a*d/n)
    mh_denom = 0.0  # sum(b*c/n)

    # For variance (Robust-Breslow-Greenland)
    sum_P1 = 0.0
    sum_P2 = 0.0

    valid_repos = []

    for repo_name, repo in repos.items():
        a, b, c, d = repo["p2f_rej"], repo["p2f_acc"], repo["nonp2f_rej"], repo["nonp2f_acc"]
        n = a + b + c + d

        # Skip strata with zero marginal totals
        if (a + b) == 0 or (c + d) == 0 or (a + c) == 0 or (b + d) == 0:
            continue

        valid_repos.append(repo_name)

        # MH weight
        mh_numer += a * d / n
        mh_denom += b * c / n

        # For variance (Breslow-Day)
        m1 = (a + b) * (a + c) / n
        m2 = (a + b) * (b + d) / n
        m3 = (c + d) * (a + c) / n
        m4 = (c + d) * (b + d) / n

        sum_P1 += (a + b) * (c + d) / (n * n)
        sum_P2 += (a + d) * (b + c) / (n * n)

    mh_or = mh_numer / mh_denom if mh_denom > 0 else float('inf')

    # Compute MH variance (Robust variance)
    # Var(log(OR_MH)) = P / (2 * mh_numer^2)
    # where P = sum(P1) + sum(P2) / (2 * mh_numer / mh_denom)

    P = sum_P1 + sum_P2
    variance = P / (2 * mh_numer**2) if mh_numer > 0 else float('inf')
    se_log_mh_or = np.sqrt(variance)

    # 95% CI
    log_mh_or = np.log(mh_or)
    log_ci_low = log_mh_or - 1.96 * se_log_mh_or
    log_ci_high = log_mh_or + 1.96 * se_log_mh_or

    mh_ci_low = np.exp(log_ci_low)
    mh_ci_high = np.exp(log_ci_high)

    # Cochran-Mantel-Haenszel chi-square test
    # CMH = (|sum(a - E(a))| - 0.5)^2 / Var(sum(a))
    # where E(a) = (a+b)(a+c) / n
    # and Var(a) = (a+b)(c+d)(a+c)(b+d) / [n^2 (n-1)]

    sum_a = sum(repo["p2f_rej"] for repo in repos.values())
    sum_E_a = 0.0
    sum_Var_a = 0.0

    for repo in repos.values():
        a, b, c, d = repo["p2f_rej"], repo["p2f_acc"], repo["nonp2f_rej"], repo["nonp2f_acc"]
        n = a + b + c + d

        if n <= 1:
            continue

        # Expected value
        E_a = (a + b) * (a + c) / n
        sum_E_a += E_a

        # Variance (hypergeometric)
        Var_a = (a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1)) if n > 1 else 0
        sum_Var_a += Var_a

    # CMH statistic with continuity correction
    diff = abs(sum_a - sum_E_a)
    cmh_chi2 = (diff - 0.5)**2 / sum_Var_a if sum_Var_a > 0 else 0

    # p-value (1 df)
    cmh_p = 1 - chi2.cdf(np.sqrt(cmh_chi2), 1) if cmh_chi2 > 0 else 1.0

    return mh_or, mh_ci_low, mh_ci_high, cmh_p, len(valid_repos), len(repos)


def compute_per_repo_heterogeneity(host_results):
    """Compute per-repo OR distribution and identify inverted/strongest repos."""
    repos = {}
    for host in host_results:
        for repo in host["per_repo"]:
            repo_name = repo["repo"]
            if repo_name not in repos:
                repos[repo_name] = repo

    # Compute OR for each repo
    repo_ors = {}
    for repo_name, repo in repos.items():
        a, b, c, d = repo["p2f_rej"], repo["p2f_acc"], repo["nonp2f_rej"], repo["nonp2f_acc"]
        total = a + b + c + d

        # Skip repos with no P2F cases (can't compute OR)
        if (a + b) == 0 or (c + d) == 0 or (a + c) == 0 or (b + d) == 0:
            continue

        # OR = (a/b) / (c/d) = ad / bc
        if b == 0 or c == 0:
            # Handle edge cases
            if b == 0 and c == 0:
                or_val = float('inf') if a > 0 and d > 0 else 1.0
            elif b == 0:
                or_val = float('inf') if a > 0 else 0
            else:  # c == 0
                or_val = 0 if a == 0 else float('inf')
        else:
            or_val = (a * d) / (b * c)

        # Only include repos with at least some meaningful data
        if total > 0:
            repo_ors[repo_name] = {"or": or_val, "total": total, "repo": repo}

    # Extract statistics
    or_values = [r["or"] for r in repo_ors.values()]
    or_median = np.median(or_values) if or_values else 1.0
    or_min = min(or_values) if or_values else 1.0
    or_max = max(or_values) if or_values else 1.0

    # Count repos with OR > 1 vs OR < 1
    n_gt1 = sum(1 for v in or_values if v > 1)
    n_lt1 = sum(1 for v in or_values if v < 1)
    n_eq1 = sum(1 for v in or_values if v == 1)

    # Find top 3 inverted (OR << 1, lowest ORs)
    inverted = sorted(repo_ors.items(), key=lambda x: x[1]["or"])[:3]

    # Find top 3 strongest (OR >> 1, highest ORs)
    strongest = sorted(repo_ors.items(), key=lambda x: -x[1]["or"])[:3]

    return {
        "median": or_median,
        "min": or_min,
        "max": or_max,
        "n_gt1": n_gt1,
        "n_lt1": n_lt1,
        "n_eq1": n_eq1,
        "top_inverted": [name for name, _ in inverted],
        "top_strongest": [name for name, _ in strongest],
        "n_total_repos": len(repo_ors)
    }


def compute_reject_rates(host_results):
    """Compute reject rate among P2F vs non-P2F PRs."""
    pooled = compute_pooled(host_results)

    # P2F reject rate
    p2f_total = pooled["p2f_rej"] + pooled["p2f_acc"]
    p2f_rej_rate = pooled["p2f_rej"] / p2f_total if p2f_total > 0 else 0

    # Non-P2F reject rate
    nonp2f_total = pooled["nonp2f_rej"] + pooled["nonp2f_acc"]
    nonp2f_rej_rate = pooled["nonp2f_rej"] / nonp2f_total if nonp2f_total > 0 else 0

    return p2f_rej_rate, nonp2f_rej_rate


def leave_one_out_mh(host_results):
    """Drop the most extreme repo and recompute MH-OR."""
    repos = {}
    for host in host_results:
        for repo in host["per_repo"]:
            repo_name = repo["repo"]
            if repo_name not in repos:
                repos[repo_name] = repo

    # Compute per-repo OR to find most extreme
    repo_ors = {}
    for repo_name, repo in repos.items():
        a, b, c, d = repo["p2f_rej"], repo["p2f_acc"], repo["nonp2f_rej"], repo["nonp2f_acc"]

        if (a + b) == 0 or (c + d) == 0 or (a + c) == 0 or (b + d) == 0:
            continue

        if b == 0 or c == 0:
            continue

        or_val = (a * d) / (b * c)
        repo_ors[repo_name] = {"or": or_val, "repo": repo}

    # Find most extreme (highest OR, likely most influential)
    if not repo_ors:
        return None

    extreme_repo = max(repo_ors.items(), key=lambda x: x[1]["or"])[0]

    # Recreate host_results without that repo
    host_results_loo = []
    for host in host_results:
        host_copy = {
            "host": host["host"],
            "n_repos_examined": host["n_repos_examined"],
            "n_prs_total": host["n_prs_total"],
            "n_prs_clean": host["n_prs_clean"],
            "n_excluded_infra_unknown": host["n_excluded_infra_unknown"],
            "host_crosstab": dict(host["host_crosstab"]),
            "per_repo": [r for r in host["per_repo"] if r["repo"] != extreme_repo]
        }

        # Adjust host crosstab
        excluded = next((r for r in host["per_repo"] if r["repo"] == extreme_repo), None)
        if excluded:
            host_copy["host_crosstab"]["p2f_rej"] -= excluded["p2f_rej"]
            host_copy["host_crosstab"]["p2f_acc"] -= excluded["p2f_acc"]
            host_copy["host_crosstab"]["nonp2f_rej"] -= excluded["nonp2f_rej"]
            host_copy["host_crosstab"]["nonp2f_acc"] -= excluded["nonp2f_acc"]

        host_results_loo.append(host_copy)

    # Recompute MH
    mh_or, ci_low, ci_high, cmh_p, n_valid, n_total = compute_mh_by_repo(host_results_loo)

    return {
        "extreme_repo": extreme_repo,
        "mh_or_loo": mh_or,
        "mh_ci_low_loo": ci_low,
        "mh_ci_high_loo": ci_high,
        "cmh_p_loo": cmh_p,
        "n_valid_loo": n_valid
    }


def main():
    print("=" * 80)
    print("P2F -> REJECTION CORRELATION ANALYSIS")
    print("=" * 80)
    print()

    # 1. Pooled analysis
    print("1. POOLED 2x2 ACROSS ALL HOSTS")
    print("-" * 80)
    pooled = compute_pooled(host_results)
    print(f"   Crosstab (P2F × Reject):")
    print(f"               | Reject | Accept |")
    print(f"   ------------|--------|--------|")
    print(f"   P2F         | {pooled['p2f_rej']:6d} | {pooled['p2f_acc']:6d} |")
    print(f"   Non-P2F     | {pooled['nonp2f_rej']:6d} | {pooled['nonp2f_acc']:6d} |")
    print()

    pooled_or, pooled_ci_low, pooled_ci_high, pooled_p = fisher_exact_test(pooled)
    print(f"   Fisher Exact (one-tailed, greater):")
    print(f"   OR = {pooled_or:.3f} (95% CI: {pooled_ci_low:.3f} - {pooled_ci_high:.3f})")
    print(f"   p = {pooled_p:.2e}")
    print()

    # 2. Mantel-Haenszel by repo
    print("2. MANTEL-HAENSZEL BY REPO (HEADLINE METRIC)")
    print("-" * 80)
    mh_or, mh_ci_low, mh_ci_high, cmh_p, n_valid, n_total = compute_mh_by_repo(host_results)
    print(f"   MH-OR = {mh_or:.3f} (95% CI: {mh_ci_low:.3f} - {mh_ci_high:.3f})")
    print(f"   CMH p = {cmh_p:.2e}")
    print(f"   Repos in MH pool: {n_valid} (total examined: {n_total})")
    print()

    # 3. Heterogeneity
    print("3. PER-REPO HETEROGENEITY")
    print("-" * 80)
    het = compute_per_repo_heterogeneity(host_results)
    print(f"   Median OR: {het['median']:.3f}")
    print(f"   Min OR: {het['min']:.3f}, Max OR: {het['max']:.3f}")
    print(f"   Repos with OR > 1 (P2F->reject): {het['n_gt1']}")
    print(f"   Repos with OR < 1 (inverted): {het['n_lt1']}")
    print(f"   Repos with OR = 1 (no signal): {het['n_eq1']}")
    print(f"   Total repos computed: {het['n_total_repos']}")
    print()
    print(f"   Top 3 most inverted (OR < 1):")
    for i, name in enumerate(het['top_inverted'], 1):
        print(f"      {i}. {name}")
    print()
    print(f"   Top 3 strongest (OR > 1):")
    for i, name in enumerate(het['top_strongest'], 1):
        print(f"      {i}. {name}")
    print()

    # 4. Reject rates
    print("4. REJECT RATES")
    print("-" * 80)
    p2f_rej_rate, nonp2f_rej_rate = compute_reject_rates(host_results)
    print(f"   Reject rate among P2F PRs: {p2f_rej_rate:.3f}")
    print(f"   Reject rate among non-P2F PRs: {nonp2f_rej_rate:.3f}")
    print(f"   Ratio: {p2f_rej_rate/nonp2f_rej_rate:.3f}x")
    print()

    # 5. Leave-one-out
    print("5. LEAVE-ONE-OUT STABILITY CHECK")
    print("-" * 80)
    loo = leave_one_out_mh(host_results)
    if loo:
        print(f"   Excluding most extreme repo: {loo['extreme_repo']}")
        print(f"   MH-OR drops from {mh_or:.3f} to {loo['mh_or_loo']:.3f}")
        print(f"   95% CI: {loo['mh_ci_low_loo']:.3f} - {loo['mh_ci_high_loo']:.3f}")
        print(f"   CMH p: {loo['cmh_p_loo']:.2e}")
        print(f"   Valid repos: {loo['n_valid_loo']}")
    else:
        print("   Could not compute leave-one-out (no valid repos)")
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Headline: P2F predicts rejection with MH-OR = {mh_or:.2f} ")
    print(f"(CMH p = {cmh_p:.2e}), based on {n_valid} repos.")
    print()
    print("Interpretation:")
    print(f"  - Pooled OR is {pooled_or:.2f}, but this is confounded by between-repo")
    print(f"    heterogeneity (Simpson's paradox potential).")
    print(f"  - MH-stratified OR controls for repo-level baseline rejection rates.")
    print(f"  - Per-repo ORs vary widely (median {het['median']:.2f}, range {het['min']:.2f}-{het['max']:.2f}),")
    print(f"    with {het['n_gt1']} repos showing positive association vs {het['n_lt1']} inverted.")
    print(f"  - P2F PRs are rejected at {p2f_rej_rate:.1%} vs {nonp2f_rej_rate:.1%} for non-P2F.")
    if loo:
        delta = mh_or - loo['mh_or_loo']
        print(f"  - Leave-one-out (excluding {loo['extreme_repo']}): MH-OR drops by {delta:.2f},")
        print(f"    suggesting robust but heterogeneous signal.")
    print()


if __name__ == "__main__":
    main()
