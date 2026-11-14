window.BENCHMARK_DATA = {
  "lastUpdate": 1722817705683,
  "repoUrl": "https://github.com/LuxDL/Lux.jl",
  "entries": {
    "Benchmark Results": [
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "avik.pal.2017@gmail.com",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "distinct": true,
          "id": "c903992547a81dc6d2bdeaac657574c991ba5d67",
          "message": "perf: fix enzyme benchmarks",
          "timestamp": "2024-07-23T13:21:52-07:00",
          "tree_id": "ddc71f0e3ee61a9754c4ba3b2d11e13b9b916f4e",
          "url": "https://github.com/LuxDL/Lux.jl/commit/c903992547a81dc6d2bdeaac657574c991ba5d67"
        },
        "date": 1721770532652,
        "tool": "julia",
        "benches": [
          {
            "name": "Dense(2 => 2)/cpu/reverse/ReverseDiff (compiled)/(2, 128)",
            "value": 3694.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Zygote/(2, 128)",
            "value": 7386.714285714285,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6016\nallocs=42\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Tracker/(2, 128)",
            "value": 20659,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17224\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/ReverseDiff/(2, 128)",
            "value": 9768.4,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9968\nallocs=59\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":5,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Flux/(2, 128)",
            "value": 8888.7,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5472\nallocs=28\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":5,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/SimpleChains/(2, 128)",
            "value": 4470.875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3120\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Enzyme/(2, 128)",
            "value": 4682.625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2320\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/NamedTuple/(2, 128)",
            "value": 1103.671875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1168\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":160,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/ComponentArray/(2, 128)",
            "value": 1186.351145038168,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1168\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":131,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/Flux/(2, 128)",
            "value": 1776.8823529411766,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2176\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":51,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/SimpleChains/(2, 128)",
            "value": 179.70380818053596,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":709,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/ReverseDiff (compiled)/(20, 128)",
            "value": 17422,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10592\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Zygote/(20, 128)",
            "value": 17012,
            "unit": "ns",
            "extra": "gctime=0\nmemory=35664\nallocs=42\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Tracker/(20, 128)",
            "value": 37251,
            "unit": "ns",
            "extra": "gctime=0\nmemory=124696\nallocs=135\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/ReverseDiff/(20, 128)",
            "value": 29145,
            "unit": "ns",
            "extra": "gctime=0\nmemory=78432\nallocs=57\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Flux/(20, 128)",
            "value": 20058,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44400\nallocs=28\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/SimpleChains/(20, 128)",
            "value": 17483,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17632\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Enzyme/(20, 128)",
            "value": 25438,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20880\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/NamedTuple/(20, 128)",
            "value": 3887.25,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10448\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/ComponentArray/(20, 128)",
            "value": 3965,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10448\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/Flux/(20, 128)",
            "value": 4967.857142857143,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20736\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/SimpleChains/(20, 128)",
            "value": 1655.1,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":10,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 3, 128)",
            "value": 38860153,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13063408\nallocs=87\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Zygote/(64, 64, 3, 128)",
            "value": 58591525.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20187840\nallocs=131\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Tracker/(64, 64, 3, 128)",
            "value": 77243718,
            "unit": "ns",
            "extra": "gctime=0\nmemory=68205704\nallocs=301\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/ReverseDiff/(64, 64, 3, 128)",
            "value": 89813513,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44197424\nallocs=197\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Flux/(64, 64, 3, 128)",
            "value": 69964192.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=26098864\nallocs=155\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/SimpleChains/(64, 64, 3, 128)",
            "value": 12192813,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5908032\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Enzyme/(64, 64, 3, 128)",
            "value": 92566221,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20188592\nallocs=139\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/NamedTuple/(64, 64, 3, 128)",
            "value": 7747332.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6739264\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/ComponentArray/(64, 64, 3, 128)",
            "value": 7636139,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6739280\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/Flux/(64, 64, 3, 128)",
            "value": 10090932,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12643680\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/SimpleChains/(64, 64, 3, 128)",
            "value": 6409069.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 16)",
            "value": 707833076,
            "unit": "ns",
            "extra": "gctime=0\nmemory=353327712\nallocs=4003\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 64)",
            "value": 2590098238,
            "unit": "ns",
            "extra": "gctime=59349731\nmemory=906193312\nallocs=4003\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 2)",
            "value": 141928214,
            "unit": "ns",
            "extra": "gctime=0\nmemory=192077392\nallocs=3969\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 16)",
            "value": 831399760,
            "unit": "ns",
            "extra": "gctime=77354605\nmemory=659567936\nallocs=10888\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 64)",
            "value": 3031770120,
            "unit": "ns",
            "extra": "gctime=354114938\nmemory=1324513984\nallocs=10889\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 2)",
            "value": 230519426,
            "unit": "ns",
            "extra": "gctime=30549382\nmemory=465628368\nallocs=10848\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 16)",
            "value": 685293775,
            "unit": "ns",
            "extra": "gctime=2413905.5\nmemory=390042496\nallocs=10658\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 64)",
            "value": 2436741285,
            "unit": "ns",
            "extra": "gctime=62173387\nmemory=1052224096\nallocs=10658\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 2)",
            "value": 128440598,
            "unit": "ns",
            "extra": "gctime=0\nmemory=196908704\nallocs=10618\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 16)",
            "value": 176803514.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=51373296\nallocs=977\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 64)",
            "value": 664156553.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=165112976\nallocs=977\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 2)",
            "value": 45698849,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18199712\nallocs=969\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 16)",
            "value": 167708912,
            "unit": "ns",
            "extra": "gctime=0\nmemory=51378224\nallocs=1022\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 64)",
            "value": 655353797,
            "unit": "ns",
            "extra": "gctime=0\nmemory=165117904\nallocs=1022\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 2)",
            "value": 30695339,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18204640\nallocs=1014\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 16)",
            "value": 187370588,
            "unit": "ns",
            "extra": "gctime=0\nmemory=69608176\nallocs=968\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 64)",
            "value": 730398133,
            "unit": "ns",
            "extra": "gctime=0\nmemory=238006832\nallocs=968\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 2)",
            "value": 38162204.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20492560\nallocs=957\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 64, 128)",
            "value": 1272794204,
            "unit": "ns",
            "extra": "gctime=175053684\nmemory=278646368\nallocs=88\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Zygote/(64, 64, 64, 128)",
            "value": 1891207049,
            "unit": "ns",
            "extra": "gctime=3145727.5\nmemory=430580128\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Tracker/(64, 64, 64, 128)",
            "value": 2377396310,
            "unit": "ns",
            "extra": "gctime=415626121\nmemory=1455080072\nallocs=302\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/ReverseDiff/(64, 64, 64, 128)",
            "value": 2412275494,
            "unit": "ns",
            "extra": "gctime=343359316\nmemory=942830496\nallocs=197\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Flux/(64, 64, 64, 128)",
            "value": 1854467971,
            "unit": "ns",
            "extra": "gctime=2684618\nmemory=556546960\nallocs=156\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Enzyme/(64, 64, 64, 128)",
            "value": 2076822407,
            "unit": "ns",
            "extra": "gctime=0\nmemory=430580640\nallocs=140\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/NamedTuple/(64, 64, 64, 128)",
            "value": 340573567.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=143677888\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/ComponentArray/(64, 64, 64, 128)",
            "value": 333252236,
            "unit": "ns",
            "extra": "gctime=0\nmemory=143677904\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/Flux/(64, 64, 64, 128)",
            "value": 459555023,
            "unit": "ns",
            "extra": "gctime=0\nmemory=269638112\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 1, 128)",
            "value": 11956140,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4360336\nallocs=87\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Zygote/(64, 64, 1, 128)",
            "value": 18159068,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6736912\nallocs=131\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Tracker/(64, 64, 1, 128)",
            "value": 19333664,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22747992\nallocs=299\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/ReverseDiff/(64, 64, 1, 128)",
            "value": 24033998,
            "unit": "ns",
            "extra": "gctime=0\nmemory=14743104\nallocs=195\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Flux/(64, 64, 1, 128)",
            "value": 17981126,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8711680\nallocs=155\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/SimpleChains/(64, 64, 1, 128)",
            "value": 1161262,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1970896\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Enzyme/(64, 64, 1, 128)",
            "value": 23276535.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6737680\nallocs=139\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/NamedTuple/(64, 64, 1, 128)",
            "value": 2359332,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2249472\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/ComponentArray/(64, 64, 1, 128)",
            "value": 2307143,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2249488\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/Flux/(64, 64, 1, 128)",
            "value": 2093660.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4217632\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/SimpleChains/(64, 64, 1, 128)",
            "value": 203713,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/ReverseDiff (compiled)/(200, 128)",
            "value": 296287,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102672\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Zygote/(200, 128)",
            "value": 267995,
            "unit": "ns",
            "extra": "gctime=0\nmemory=470896\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Tracker/(200, 128)",
            "value": 371109,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1614552\nallocs=149\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/ReverseDiff/(200, 128)",
            "value": 412547,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1040224\nallocs=66\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Flux/(200, 128)",
            "value": 276370.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=571712\nallocs=33\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/SimpleChains/(200, 128)",
            "value": 410583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=747872\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Enzyme/(200, 128)",
            "value": 398600,
            "unit": "ns",
            "extra": "gctime=0\nmemory=205040\nallocs=7\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/NamedTuple/(200, 128)",
            "value": 84133.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102528\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/ComponentArray/(200, 128)",
            "value": 87244,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102528\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/Flux/(200, 128)",
            "value": 87415,
            "unit": "ns",
            "extra": "gctime=0\nmemory=204896\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/SimpleChains/(200, 128)",
            "value": 104687,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 16, 128)",
            "value": 191531913,
            "unit": "ns",
            "extra": "gctime=0\nmemory=69640608\nallocs=87\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Zygote/(64, 64, 16, 128)",
            "value": 327882868,
            "unit": "ns",
            "extra": "gctime=0\nmemory=107626016\nallocs=131\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Tracker/(64, 64, 16, 128)",
            "value": 422399750.5,
            "unit": "ns",
            "extra": "gctime=1910402.5\nmemory=363701768\nallocs=299\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/ReverseDiff/(64, 64, 16, 128)",
            "value": 483565853,
            "unit": "ns",
            "extra": "gctime=0\nmemory=235664480\nallocs=195\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Flux/(64, 64, 16, 128)",
            "value": 387480865,
            "unit": "ns",
            "extra": "gctime=0\nmemory=139122704\nallocs=155\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/SimpleChains/(64, 64, 16, 128)",
            "value": 337771969,
            "unit": "ns",
            "extra": "gctime=0\nmemory=31520784\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Enzyme/(64, 64, 16, 128)",
            "value": 480545110.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=107626720\nallocs=139\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/NamedTuple/(64, 64, 16, 128)",
            "value": 47647427,
            "unit": "ns",
            "extra": "gctime=0\nmemory=35922880\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/ComponentArray/(64, 64, 16, 128)",
            "value": 47230335,
            "unit": "ns",
            "extra": "gctime=0\nmemory=35922896\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/Flux/(64, 64, 16, 128)",
            "value": 50071719.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=67412960\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/SimpleChains/(64, 64, 16, 128)",
            "value": 28845102,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/ReverseDiff (compiled)/(2000, 128)",
            "value": 19018583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024272\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Zygote/(2000, 128)",
            "value": 19702711,
            "unit": "ns",
            "extra": "gctime=0\nmemory=19082928\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Tracker/(2000, 128)",
            "value": 23539011.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=59293848\nallocs=149\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/ReverseDiff/(2000, 128)",
            "value": 24139841.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=39178656\nallocs=66\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Flux/(2000, 128)",
            "value": 19766095,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20105344\nallocs=33\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Enzyme/(2000, 128)",
            "value": 21127350.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2048240\nallocs=7\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/NamedTuple/(2000, 128)",
            "value": 6590543,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024128\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/ComponentArray/(2000, 128)",
            "value": 6541563,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024128\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/Flux/(2000, 128)",
            "value": 6538585,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2048096\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "avik.pal.2017@gmail.com",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "distinct": true,
          "id": "8d2e585dc0508bd24da2fca7ea5381272261a93a",
          "message": "test: trigger enzyme tests",
          "timestamp": "2024-07-24T08:09:20-07:00",
          "tree_id": "9da4eccd262ba9c2f55dc46100e9a21b38b7d999",
          "url": "https://github.com/LuxDL/Lux.jl/commit/8d2e585dc0508bd24da2fca7ea5381272261a93a"
        },
        "date": 1721836032780,
        "tool": "julia",
        "benches": [
          {
            "name": "Dense(2 => 2)/cpu/reverse/ReverseDiff (compiled)/(2, 128)",
            "value": 3898.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Zygote/(2, 128)",
            "value": 7307.6,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6016\nallocs=42\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":5,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Tracker/(2, 128)",
            "value": 20728,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17224\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/ReverseDiff/(2, 128)",
            "value": 9790.2,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9968\nallocs=59\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":5,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Flux/(2, 128)",
            "value": 9184.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5472\nallocs=28\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":4,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/SimpleChains/(2, 128)",
            "value": 4464.625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3120\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Enzyme/(2, 128)",
            "value": 4689.875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2320\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/NamedTuple/(2, 128)",
            "value": 1119.3288590604027,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1168\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":149,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/ComponentArray/(2, 128)",
            "value": 1186.4705882352941,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1168\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":136,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/Flux/(2, 128)",
            "value": 1781.9122807017543,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2176\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":57,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/SimpleChains/(2, 128)",
            "value": 179.221140472879,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":719,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/ReverseDiff (compiled)/(20, 128)",
            "value": 17212,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10592\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Zygote/(20, 128)",
            "value": 16962,
            "unit": "ns",
            "extra": "gctime=0\nmemory=35664\nallocs=42\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Tracker/(20, 128)",
            "value": 37089,
            "unit": "ns",
            "extra": "gctime=0\nmemory=124696\nallocs=135\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/ReverseDiff/(20, 128)",
            "value": 29054,
            "unit": "ns",
            "extra": "gctime=0\nmemory=78432\nallocs=57\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Flux/(20, 128)",
            "value": 21500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44400\nallocs=28\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/SimpleChains/(20, 128)",
            "value": 17192,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17632\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Enzyme/(20, 128)",
            "value": 25507,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20880\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/NamedTuple/(20, 128)",
            "value": 3858.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10448\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/ComponentArray/(20, 128)",
            "value": 3944.875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10448\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/Flux/(20, 128)",
            "value": 4901.857142857143,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20736\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/SimpleChains/(20, 128)",
            "value": 1653.1,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":10,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 3, 128)",
            "value": 39417430,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13063408\nallocs=87\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Zygote/(64, 64, 3, 128)",
            "value": 59026753,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20187840\nallocs=131\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Tracker/(64, 64, 3, 128)",
            "value": 78753982.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=68205704\nallocs=301\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/ReverseDiff/(64, 64, 3, 128)",
            "value": 91577217,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44197424\nallocs=197\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Flux/(64, 64, 3, 128)",
            "value": 75064628,
            "unit": "ns",
            "extra": "gctime=0\nmemory=26098864\nallocs=155\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/SimpleChains/(64, 64, 3, 128)",
            "value": 12478072,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5908032\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Enzyme/(64, 64, 3, 128)",
            "value": 88187659,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20188592\nallocs=139\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/NamedTuple/(64, 64, 3, 128)",
            "value": 7734313,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6739264\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/ComponentArray/(64, 64, 3, 128)",
            "value": 7616145,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6739280\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/Flux/(64, 64, 3, 128)",
            "value": 10169419,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12643680\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/SimpleChains/(64, 64, 3, 128)",
            "value": 6475298.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 16)",
            "value": 714601212,
            "unit": "ns",
            "extra": "gctime=0\nmemory=353327712\nallocs=4003\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 64)",
            "value": 2575011212,
            "unit": "ns",
            "extra": "gctime=55583347\nmemory=906193312\nallocs=4003\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 2)",
            "value": 148434028,
            "unit": "ns",
            "extra": "gctime=0\nmemory=192077392\nallocs=3969\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 16)",
            "value": 835656592,
            "unit": "ns",
            "extra": "gctime=62499455\nmemory=659567936\nallocs=10888\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 64)",
            "value": 2880099801,
            "unit": "ns",
            "extra": "gctime=383114039\nmemory=1324513984\nallocs=10889\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 2)",
            "value": 230296223,
            "unit": "ns",
            "extra": "gctime=29036666\nmemory=465628368\nallocs=10848\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 16)",
            "value": 754723456,
            "unit": "ns",
            "extra": "gctime=3228425\nmemory=390042496\nallocs=10658\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 64)",
            "value": 2620964530,
            "unit": "ns",
            "extra": "gctime=57779463\nmemory=1052224096\nallocs=10658\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 2)",
            "value": 137992622,
            "unit": "ns",
            "extra": "gctime=0\nmemory=196908704\nallocs=10618\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 16)",
            "value": 177381390,
            "unit": "ns",
            "extra": "gctime=0\nmemory=51373296\nallocs=977\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 64)",
            "value": 664563096,
            "unit": "ns",
            "extra": "gctime=0\nmemory=165112976\nallocs=977\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 2)",
            "value": 35498073.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18199712\nallocs=969\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 16)",
            "value": 167948188,
            "unit": "ns",
            "extra": "gctime=0\nmemory=51378224\nallocs=1022\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 64)",
            "value": 655194974,
            "unit": "ns",
            "extra": "gctime=0\nmemory=165117904\nallocs=1022\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 2)",
            "value": 30691371.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18204640\nallocs=1014\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 16)",
            "value": 190361878,
            "unit": "ns",
            "extra": "gctime=0\nmemory=69608176\nallocs=968\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 64)",
            "value": 742561584.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=238006832\nallocs=968\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 2)",
            "value": 37831252,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20492560\nallocs=957\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 64, 128)",
            "value": 1299516480,
            "unit": "ns",
            "extra": "gctime=199654126.5\nmemory=278646368\nallocs=88\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Zygote/(64, 64, 64, 128)",
            "value": 1906477766,
            "unit": "ns",
            "extra": "gctime=3754132.5\nmemory=430580128\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Tracker/(64, 64, 64, 128)",
            "value": 2299658099,
            "unit": "ns",
            "extra": "gctime=417573419\nmemory=1455080072\nallocs=302\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/ReverseDiff/(64, 64, 64, 128)",
            "value": 2458436737,
            "unit": "ns",
            "extra": "gctime=354708842\nmemory=942830496\nallocs=197\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Flux/(64, 64, 64, 128)",
            "value": 1945065634.5,
            "unit": "ns",
            "extra": "gctime=1766162\nmemory=556546960\nallocs=156\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Enzyme/(64, 64, 64, 128)",
            "value": 2069602165,
            "unit": "ns",
            "extra": "gctime=0\nmemory=430580640\nallocs=140\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/NamedTuple/(64, 64, 64, 128)",
            "value": 344998452,
            "unit": "ns",
            "extra": "gctime=0\nmemory=143677888\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/ComponentArray/(64, 64, 64, 128)",
            "value": 346311953.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=143677904\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/Flux/(64, 64, 64, 128)",
            "value": 379997474,
            "unit": "ns",
            "extra": "gctime=0\nmemory=269638112\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 1, 128)",
            "value": 12056841,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4360336\nallocs=87\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Zygote/(64, 64, 1, 128)",
            "value": 18191891,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6736912\nallocs=131\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Tracker/(64, 64, 1, 128)",
            "value": 19288745,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22747992\nallocs=299\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/ReverseDiff/(64, 64, 1, 128)",
            "value": 23960447,
            "unit": "ns",
            "extra": "gctime=0\nmemory=14743104\nallocs=195\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Flux/(64, 64, 1, 128)",
            "value": 18001949,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8711680\nallocs=155\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/SimpleChains/(64, 64, 1, 128)",
            "value": 1175012,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1970896\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Enzyme/(64, 64, 1, 128)",
            "value": 23334989,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6737680\nallocs=139\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/NamedTuple/(64, 64, 1, 128)",
            "value": 2293583.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2249472\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/ComponentArray/(64, 64, 1, 128)",
            "value": 2214902,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2249488\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/Flux/(64, 64, 1, 128)",
            "value": 2070653,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4217632\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/SimpleChains/(64, 64, 1, 128)",
            "value": 207386,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/ReverseDiff (compiled)/(200, 128)",
            "value": 295220,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102672\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Zygote/(200, 128)",
            "value": 269611,
            "unit": "ns",
            "extra": "gctime=0\nmemory=470896\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Tracker/(200, 128)",
            "value": 368927,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1614552\nallocs=149\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/ReverseDiff/(200, 128)",
            "value": 411035,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1040224\nallocs=66\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Flux/(200, 128)",
            "value": 278338,
            "unit": "ns",
            "extra": "gctime=0\nmemory=571712\nallocs=33\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/SimpleChains/(200, 128)",
            "value": 408440,
            "unit": "ns",
            "extra": "gctime=0\nmemory=747872\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Enzyme/(200, 128)",
            "value": 397870,
            "unit": "ns",
            "extra": "gctime=0\nmemory=205040\nallocs=7\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/NamedTuple/(200, 128)",
            "value": 81491,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102528\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/ComponentArray/(200, 128)",
            "value": 81742,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102528\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/Flux/(200, 128)",
            "value": 87142.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=204896\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/SimpleChains/(200, 128)",
            "value": 104725,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 16, 128)",
            "value": 189895146,
            "unit": "ns",
            "extra": "gctime=0\nmemory=69640608\nallocs=87\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Zygote/(64, 64, 16, 128)",
            "value": 330378973,
            "unit": "ns",
            "extra": "gctime=0\nmemory=107626016\nallocs=131\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Tracker/(64, 64, 16, 128)",
            "value": 425406175,
            "unit": "ns",
            "extra": "gctime=1877349\nmemory=363701768\nallocs=299\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/ReverseDiff/(64, 64, 16, 128)",
            "value": 484809766,
            "unit": "ns",
            "extra": "gctime=0\nmemory=235664480\nallocs=195\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Flux/(64, 64, 16, 128)",
            "value": 357948899,
            "unit": "ns",
            "extra": "gctime=0\nmemory=139122704\nallocs=155\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/SimpleChains/(64, 64, 16, 128)",
            "value": 344315432,
            "unit": "ns",
            "extra": "gctime=0\nmemory=31520784\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Enzyme/(64, 64, 16, 128)",
            "value": 449554706,
            "unit": "ns",
            "extra": "gctime=0\nmemory=107626720\nallocs=139\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/NamedTuple/(64, 64, 16, 128)",
            "value": 47365748,
            "unit": "ns",
            "extra": "gctime=0\nmemory=35922880\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/ComponentArray/(64, 64, 16, 128)",
            "value": 46685949.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=35922896\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/Flux/(64, 64, 16, 128)",
            "value": 59744200,
            "unit": "ns",
            "extra": "gctime=0\nmemory=67412960\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/SimpleChains/(64, 64, 16, 128)",
            "value": 28213801,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/ReverseDiff (compiled)/(2000, 128)",
            "value": 19193456.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024272\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Zygote/(2000, 128)",
            "value": 19696227,
            "unit": "ns",
            "extra": "gctime=0\nmemory=19082928\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Tracker/(2000, 128)",
            "value": 23742308,
            "unit": "ns",
            "extra": "gctime=0\nmemory=59293848\nallocs=149\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/ReverseDiff/(2000, 128)",
            "value": 24224045,
            "unit": "ns",
            "extra": "gctime=0\nmemory=39178656\nallocs=66\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Flux/(2000, 128)",
            "value": 19666246.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20105344\nallocs=33\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Enzyme/(2000, 128)",
            "value": 20955027,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2048240\nallocs=7\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/NamedTuple/(2000, 128)",
            "value": 6545190,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024128\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/ComponentArray/(2000, 128)",
            "value": 6556441,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024128\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/Flux/(2000, 128)",
            "value": 6537709,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2048096\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "41898282+github-actions[bot]@users.noreply.github.com",
            "name": "github-actions[bot]",
            "username": "github-actions[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "708c10cc1f09f8a8a1eb966dde6521664a7502d1",
          "message": "chore: bump compat for WeightInitializers to 1, (keep existing compat) (#791)\n\nCo-authored-by: CompatHelper Julia <compathelper_noreply@julialang.org>",
          "timestamp": "2024-07-25T18:34:49-07:00",
          "tree_id": "c016624c2f2878e198648ef71f004588eb8489d8",
          "url": "https://github.com/LuxDL/Lux.jl/commit/708c10cc1f09f8a8a1eb966dde6521664a7502d1"
        },
        "date": 1721961663398,
        "tool": "julia",
        "benches": [
          {
            "name": "Dense(2 => 2)/cpu/reverse/ReverseDiff (compiled)/(2, 128)",
            "value": 3663.125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Zygote/(2, 128)",
            "value": 7402.428571428572,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6016\nallocs=42\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Tracker/(2, 128)",
            "value": 21370,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17224\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/ReverseDiff/(2, 128)",
            "value": 9748.2,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9968\nallocs=59\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":5,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Flux/(2, 128)",
            "value": 8969.25,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5472\nallocs=28\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":4,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/SimpleChains/(2, 128)",
            "value": 4458.375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3120\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Enzyme/(2, 128)",
            "value": 4643.625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2320\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/NamedTuple/(2, 128)",
            "value": 1171.4923076923078,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1168\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":130,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/ComponentArray/(2, 128)",
            "value": 1197.5373134328358,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1168\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":134,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/Flux/(2, 128)",
            "value": 1840.7045454545455,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2176\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":44,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/SimpleChains/(2, 128)",
            "value": 179.47899159663865,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":714,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/ReverseDiff (compiled)/(20, 128)",
            "value": 17282.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10592\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Zygote/(20, 128)",
            "value": 17142,
            "unit": "ns",
            "extra": "gctime=0\nmemory=35664\nallocs=42\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Tracker/(20, 128)",
            "value": 37821,
            "unit": "ns",
            "extra": "gctime=0\nmemory=124696\nallocs=135\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/ReverseDiff/(20, 128)",
            "value": 28674,
            "unit": "ns",
            "extra": "gctime=0\nmemory=78432\nallocs=57\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Flux/(20, 128)",
            "value": 20158,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44400\nallocs=28\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/SimpleChains/(20, 128)",
            "value": 17573,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17632\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Enzyme/(20, 128)",
            "value": 25558,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20880\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/NamedTuple/(20, 128)",
            "value": 3867.25,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10448\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/ComponentArray/(20, 128)",
            "value": 3951.125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10448\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/Flux/(20, 128)",
            "value": 4960.714285714285,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20736\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/SimpleChains/(20, 128)",
            "value": 1652.1,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":10,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 3, 128)",
            "value": 40993082,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13063408\nallocs=87\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Zygote/(64, 64, 3, 128)",
            "value": 58497971,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20187840\nallocs=131\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Tracker/(64, 64, 3, 128)",
            "value": 82819165,
            "unit": "ns",
            "extra": "gctime=0\nmemory=68205704\nallocs=301\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/ReverseDiff/(64, 64, 3, 128)",
            "value": 92585992,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44197424\nallocs=197\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Flux/(64, 64, 3, 128)",
            "value": 78243436,
            "unit": "ns",
            "extra": "gctime=0\nmemory=26098864\nallocs=155\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/SimpleChains/(64, 64, 3, 128)",
            "value": 12365359.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5908032\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Enzyme/(64, 64, 3, 128)",
            "value": 90927109,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20188592\nallocs=139\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/NamedTuple/(64, 64, 3, 128)",
            "value": 7696963,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6739264\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/ComponentArray/(64, 64, 3, 128)",
            "value": 7605182,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6739280\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/Flux/(64, 64, 3, 128)",
            "value": 11852587,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12643680\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/SimpleChains/(64, 64, 3, 128)",
            "value": 6417475.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 16)",
            "value": 715800734,
            "unit": "ns",
            "extra": "gctime=0\nmemory=353327712\nallocs=4003\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 64)",
            "value": 2585785068,
            "unit": "ns",
            "extra": "gctime=54701698\nmemory=906193312\nallocs=4003\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 2)",
            "value": 145144322,
            "unit": "ns",
            "extra": "gctime=0\nmemory=192077392\nallocs=3969\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 16)",
            "value": 857691023,
            "unit": "ns",
            "extra": "gctime=66172498\nmemory=659567936\nallocs=10888\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 64)",
            "value": 3013396972,
            "unit": "ns",
            "extra": "gctime=350410862\nmemory=1324513984\nallocs=10889\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 2)",
            "value": 271704927.5,
            "unit": "ns",
            "extra": "gctime=29071073.5\nmemory=465628368\nallocs=10848\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 16)",
            "value": 796866243,
            "unit": "ns",
            "extra": "gctime=2898288\nmemory=390042496\nallocs=10658\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 64)",
            "value": 2712566258,
            "unit": "ns",
            "extra": "gctime=54451488\nmemory=1052224096\nallocs=10658\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 2)",
            "value": 130670825,
            "unit": "ns",
            "extra": "gctime=0\nmemory=196908704\nallocs=10618\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 16)",
            "value": 176972248,
            "unit": "ns",
            "extra": "gctime=0\nmemory=51373296\nallocs=977\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 64)",
            "value": 660618124.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=165112976\nallocs=977\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 2)",
            "value": 37524620,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18199712\nallocs=969\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 16)",
            "value": 167203678,
            "unit": "ns",
            "extra": "gctime=0\nmemory=51378224\nallocs=1022\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 64)",
            "value": 650758175,
            "unit": "ns",
            "extra": "gctime=0\nmemory=165117904\nallocs=1022\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 2)",
            "value": 30535622,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18204640\nallocs=1014\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 16)",
            "value": 212791073.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=69608176\nallocs=968\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 64)",
            "value": 854087743,
            "unit": "ns",
            "extra": "gctime=0\nmemory=238006832\nallocs=968\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 2)",
            "value": 36391416,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20492560\nallocs=957\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 64, 128)",
            "value": 1349278132.5,
            "unit": "ns",
            "extra": "gctime=196373068.5\nmemory=278646368\nallocs=88\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Zygote/(64, 64, 64, 128)",
            "value": 1887134256.5,
            "unit": "ns",
            "extra": "gctime=3693798\nmemory=430580128\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Tracker/(64, 64, 64, 128)",
            "value": 2450133270,
            "unit": "ns",
            "extra": "gctime=444532214\nmemory=1455080072\nallocs=302\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/ReverseDiff/(64, 64, 64, 128)",
            "value": 2519902348,
            "unit": "ns",
            "extra": "gctime=390477427\nmemory=942830496\nallocs=197\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Flux/(64, 64, 64, 128)",
            "value": 1970487480.5,
            "unit": "ns",
            "extra": "gctime=1750727\nmemory=556546960\nallocs=156\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Enzyme/(64, 64, 64, 128)",
            "value": 2023950360,
            "unit": "ns",
            "extra": "gctime=0\nmemory=430580640\nallocs=140\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/NamedTuple/(64, 64, 64, 128)",
            "value": 347833228,
            "unit": "ns",
            "extra": "gctime=0\nmemory=143677888\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/ComponentArray/(64, 64, 64, 128)",
            "value": 342952632,
            "unit": "ns",
            "extra": "gctime=0\nmemory=143677904\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/Flux/(64, 64, 64, 128)",
            "value": 404098087,
            "unit": "ns",
            "extra": "gctime=0\nmemory=269638112\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 1, 128)",
            "value": 12035460,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4360336\nallocs=87\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Zygote/(64, 64, 1, 128)",
            "value": 18156508.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6736912\nallocs=131\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Tracker/(64, 64, 1, 128)",
            "value": 19338119,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22747992\nallocs=299\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/ReverseDiff/(64, 64, 1, 128)",
            "value": 24010708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=14743104\nallocs=195\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Flux/(64, 64, 1, 128)",
            "value": 17942213,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8711680\nallocs=155\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/SimpleChains/(64, 64, 1, 128)",
            "value": 1209037,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1970896\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Enzyme/(64, 64, 1, 128)",
            "value": 23232956,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6737680\nallocs=139\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/NamedTuple/(64, 64, 1, 128)",
            "value": 2293500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2249472\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/ComponentArray/(64, 64, 1, 128)",
            "value": 2221085,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2249488\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/Flux/(64, 64, 1, 128)",
            "value": 2088387,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4217632\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/SimpleChains/(64, 64, 1, 128)",
            "value": 210493,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/ReverseDiff (compiled)/(200, 128)",
            "value": 297254.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102672\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Zygote/(200, 128)",
            "value": 268511,
            "unit": "ns",
            "extra": "gctime=0\nmemory=470896\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Tracker/(200, 128)",
            "value": 372155,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1614552\nallocs=149\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/ReverseDiff/(200, 128)",
            "value": 414683,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1040224\nallocs=66\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Flux/(200, 128)",
            "value": 275874.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=571712\nallocs=33\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/SimpleChains/(200, 128)",
            "value": 413892,
            "unit": "ns",
            "extra": "gctime=0\nmemory=747872\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Enzyme/(200, 128)",
            "value": 399165,
            "unit": "ns",
            "extra": "gctime=0\nmemory=205040\nallocs=7\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/NamedTuple/(200, 128)",
            "value": 81923,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102528\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/ComponentArray/(200, 128)",
            "value": 82824,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102528\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/Flux/(200, 128)",
            "value": 87643,
            "unit": "ns",
            "extra": "gctime=0\nmemory=204896\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/SimpleChains/(200, 128)",
            "value": 104415,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 16, 128)",
            "value": 193631929,
            "unit": "ns",
            "extra": "gctime=0\nmemory=69640608\nallocs=87\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Zygote/(64, 64, 16, 128)",
            "value": 331527065,
            "unit": "ns",
            "extra": "gctime=0\nmemory=107626016\nallocs=131\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Tracker/(64, 64, 16, 128)",
            "value": 412143695,
            "unit": "ns",
            "extra": "gctime=1896153.5\nmemory=363701768\nallocs=299\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/ReverseDiff/(64, 64, 16, 128)",
            "value": 469727799.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=235664480\nallocs=195\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Flux/(64, 64, 16, 128)",
            "value": 384904410,
            "unit": "ns",
            "extra": "gctime=0\nmemory=139122704\nallocs=155\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/SimpleChains/(64, 64, 16, 128)",
            "value": 355709715,
            "unit": "ns",
            "extra": "gctime=0\nmemory=31520784\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Enzyme/(64, 64, 16, 128)",
            "value": 472893365,
            "unit": "ns",
            "extra": "gctime=0\nmemory=107626720\nallocs=139\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/NamedTuple/(64, 64, 16, 128)",
            "value": 47367307,
            "unit": "ns",
            "extra": "gctime=0\nmemory=35922880\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/ComponentArray/(64, 64, 16, 128)",
            "value": 46829064.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=35922896\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/Flux/(64, 64, 16, 128)",
            "value": 57797619.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=67412960\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/SimpleChains/(64, 64, 16, 128)",
            "value": 28515793,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/ReverseDiff (compiled)/(2000, 128)",
            "value": 19208519,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024272\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Zygote/(2000, 128)",
            "value": 19878820,
            "unit": "ns",
            "extra": "gctime=0\nmemory=19082928\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Tracker/(2000, 128)",
            "value": 23974472,
            "unit": "ns",
            "extra": "gctime=0\nmemory=59293848\nallocs=149\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/ReverseDiff/(2000, 128)",
            "value": 24477689.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=39178656\nallocs=66\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Flux/(2000, 128)",
            "value": 19960697.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20105344\nallocs=33\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Enzyme/(2000, 128)",
            "value": 21327622,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2048240\nallocs=7\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/NamedTuple/(2000, 128)",
            "value": 6617984,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024128\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/ComponentArray/(2000, 128)",
            "value": 6558899,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024128\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/Flux/(2000, 128)",
            "value": 6559310,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2048096\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e1594bf894ccee0967e21069f7d5df146a1cc8a7",
          "message": "docs: update docs from downstream changes (#790)\n\n* docs: make the interface specification sound mandatory\r\n\r\n* docs: add list of supported rngs for init\r\n\r\n* docs: remove packages no longer explicitly needed",
          "timestamp": "2024-07-25T19:35:44-07:00",
          "tree_id": "70d143e08d55cd84d0d506040baf4c30e79d8a80",
          "url": "https://github.com/LuxDL/Lux.jl/commit/e1594bf894ccee0967e21069f7d5df146a1cc8a7"
        },
        "date": 1721967065353,
        "tool": "julia",
        "benches": [
          {
            "name": "Dense(2 => 2)/cpu/reverse/ReverseDiff (compiled)/(2, 128)",
            "value": 3693.125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Zygote/(2, 128)",
            "value": 7382.428571428572,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6016\nallocs=42\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Tracker/(2, 128)",
            "value": 20759,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17224\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/ReverseDiff/(2, 128)",
            "value": 9812.4,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9968\nallocs=59\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":5,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Flux/(2, 128)",
            "value": 9094.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5472\nallocs=28\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":4,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/SimpleChains/(2, 128)",
            "value": 4493.375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3120\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Enzyme/(2, 128)",
            "value": 4562.25,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2320\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/NamedTuple/(2, 128)",
            "value": 1109.9668874172185,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1168\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":151,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/ComponentArray/(2, 128)",
            "value": 1181.937062937063,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1168\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":143,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/Flux/(2, 128)",
            "value": 1793.1929824561403,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2176\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":57,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/SimpleChains/(2, 128)",
            "value": 179.68289290681503,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":719,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/ReverseDiff (compiled)/(20, 128)",
            "value": 17342,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10592\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Zygote/(20, 128)",
            "value": 16932,
            "unit": "ns",
            "extra": "gctime=0\nmemory=35664\nallocs=42\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Tracker/(20, 128)",
            "value": 37520,
            "unit": "ns",
            "extra": "gctime=0\nmemory=124696\nallocs=135\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/ReverseDiff/(20, 128)",
            "value": 28584,
            "unit": "ns",
            "extra": "gctime=0\nmemory=78432\nallocs=57\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Flux/(20, 128)",
            "value": 20278,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44400\nallocs=28\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/SimpleChains/(20, 128)",
            "value": 17392.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17632\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Enzyme/(20, 128)",
            "value": 25578,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20880\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/NamedTuple/(20, 128)",
            "value": 3938.625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10448\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/ComponentArray/(20, 128)",
            "value": 3991.25,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10448\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/Flux/(20, 128)",
            "value": 5029.428571428572,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20736\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/SimpleChains/(20, 128)",
            "value": 1652.1,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":10,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 3, 128)",
            "value": 39521781,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13063408\nallocs=87\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Zygote/(64, 64, 3, 128)",
            "value": 58585547,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20187840\nallocs=131\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Tracker/(64, 64, 3, 128)",
            "value": 78099068,
            "unit": "ns",
            "extra": "gctime=0\nmemory=68205704\nallocs=301\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/ReverseDiff/(64, 64, 3, 128)",
            "value": 90769018,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44197424\nallocs=197\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Flux/(64, 64, 3, 128)",
            "value": 72690086,
            "unit": "ns",
            "extra": "gctime=0\nmemory=26098864\nallocs=155\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/SimpleChains/(64, 64, 3, 128)",
            "value": 12054967,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5908032\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Enzyme/(64, 64, 3, 128)",
            "value": 83400155.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20188592\nallocs=139\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/NamedTuple/(64, 64, 3, 128)",
            "value": 7721292,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6739264\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/ComponentArray/(64, 64, 3, 128)",
            "value": 7581495.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6739280\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/Flux/(64, 64, 3, 128)",
            "value": 9955016,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12643680\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/SimpleChains/(64, 64, 3, 128)",
            "value": 6388090.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 16)",
            "value": 725603932,
            "unit": "ns",
            "extra": "gctime=0\nmemory=353327712\nallocs=4003\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 64)",
            "value": 2550320476,
            "unit": "ns",
            "extra": "gctime=53310363\nmemory=906193312\nallocs=4003\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 2)",
            "value": 138974187,
            "unit": "ns",
            "extra": "gctime=0\nmemory=192077392\nallocs=3969\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 16)",
            "value": 803301267,
            "unit": "ns",
            "extra": "gctime=65104339\nmemory=659570112\nallocs=10889\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 64)",
            "value": 2816236020,
            "unit": "ns",
            "extra": "gctime=338452811\nmemory=1324513984\nallocs=10889\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 2)",
            "value": 235878850.5,
            "unit": "ns",
            "extra": "gctime=30788623.5\nmemory=465628368\nallocs=10848\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 16)",
            "value": 650791127,
            "unit": "ns",
            "extra": "gctime=3326063\nmemory=390042496\nallocs=10658\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 64)",
            "value": 2603380595.5,
            "unit": "ns",
            "extra": "gctime=217216546.5\nmemory=1052224096\nallocs=10658\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 2)",
            "value": 135543456,
            "unit": "ns",
            "extra": "gctime=0\nmemory=196908704\nallocs=10618\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 16)",
            "value": 175275149,
            "unit": "ns",
            "extra": "gctime=0\nmemory=51373296\nallocs=977\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 64)",
            "value": 658937990.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=165112976\nallocs=977\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 2)",
            "value": 35195315,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18199712\nallocs=969\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 16)",
            "value": 166640478.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=51378224\nallocs=1022\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 64)",
            "value": 647602016,
            "unit": "ns",
            "extra": "gctime=0\nmemory=165117904\nallocs=1022\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 2)",
            "value": 30477927,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18204640\nallocs=1014\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 16)",
            "value": 186029467,
            "unit": "ns",
            "extra": "gctime=0\nmemory=69608176\nallocs=968\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 64)",
            "value": 718031730.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=238006832\nallocs=968\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 2)",
            "value": 38445179.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20492560\nallocs=957\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 64, 128)",
            "value": 1233631654,
            "unit": "ns",
            "extra": "gctime=171532198.5\nmemory=278646368\nallocs=88\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Zygote/(64, 64, 64, 128)",
            "value": 1887704692.5,
            "unit": "ns",
            "extra": "gctime=4032826\nmemory=430580128\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Tracker/(64, 64, 64, 128)",
            "value": 2231708206,
            "unit": "ns",
            "extra": "gctime=394397897\nmemory=1455080072\nallocs=302\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/ReverseDiff/(64, 64, 64, 128)",
            "value": 2471164221,
            "unit": "ns",
            "extra": "gctime=340546627\nmemory=942830496\nallocs=197\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Flux/(64, 64, 64, 128)",
            "value": 1848957394.5,
            "unit": "ns",
            "extra": "gctime=1848428.5\nmemory=556546960\nallocs=156\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Enzyme/(64, 64, 64, 128)",
            "value": 1990760702,
            "unit": "ns",
            "extra": "gctime=0\nmemory=430580640\nallocs=140\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/NamedTuple/(64, 64, 64, 128)",
            "value": 333383541,
            "unit": "ns",
            "extra": "gctime=0\nmemory=143677888\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/ComponentArray/(64, 64, 64, 128)",
            "value": 330228531,
            "unit": "ns",
            "extra": "gctime=0\nmemory=143677904\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/Flux/(64, 64, 64, 128)",
            "value": 365192205,
            "unit": "ns",
            "extra": "gctime=0\nmemory=269638112\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 1, 128)",
            "value": 11909407,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4360336\nallocs=87\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Zygote/(64, 64, 1, 128)",
            "value": 18028312,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6736912\nallocs=131\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Tracker/(64, 64, 1, 128)",
            "value": 19050490,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22747992\nallocs=299\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/ReverseDiff/(64, 64, 1, 128)",
            "value": 23803362,
            "unit": "ns",
            "extra": "gctime=0\nmemory=14743104\nallocs=195\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Flux/(64, 64, 1, 128)",
            "value": 17829963,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8711680\nallocs=155\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/SimpleChains/(64, 64, 1, 128)",
            "value": 1163076.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1970896\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Enzyme/(64, 64, 1, 128)",
            "value": 22925657,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6737680\nallocs=139\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/NamedTuple/(64, 64, 1, 128)",
            "value": 2329735,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2249472\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/ComponentArray/(64, 64, 1, 128)",
            "value": 2215453,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2249488\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/Flux/(64, 64, 1, 128)",
            "value": 2068279,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4217632\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/SimpleChains/(64, 64, 1, 128)",
            "value": 200525.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/ReverseDiff (compiled)/(200, 128)",
            "value": 293690.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102672\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Zygote/(200, 128)",
            "value": 267291,
            "unit": "ns",
            "extra": "gctime=0\nmemory=470896\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Tracker/(200, 128)",
            "value": 369002,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1614552\nallocs=149\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/ReverseDiff/(200, 128)",
            "value": 409669,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1040224\nallocs=66\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Flux/(200, 128)",
            "value": 275557,
            "unit": "ns",
            "extra": "gctime=0\nmemory=571712\nallocs=33\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/SimpleChains/(200, 128)",
            "value": 407905,
            "unit": "ns",
            "extra": "gctime=0\nmemory=747872\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Enzyme/(200, 128)",
            "value": 397495,
            "unit": "ns",
            "extra": "gctime=0\nmemory=205040\nallocs=7\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/NamedTuple/(200, 128)",
            "value": 81824,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102528\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/ComponentArray/(200, 128)",
            "value": 82175,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102528\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/Flux/(200, 128)",
            "value": 87213,
            "unit": "ns",
            "extra": "gctime=0\nmemory=204896\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/SimpleChains/(200, 128)",
            "value": 104706,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 16, 128)",
            "value": 193496448,
            "unit": "ns",
            "extra": "gctime=0\nmemory=69640608\nallocs=87\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Zygote/(64, 64, 16, 128)",
            "value": 330361771,
            "unit": "ns",
            "extra": "gctime=0\nmemory=107626016\nallocs=131\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Tracker/(64, 64, 16, 128)",
            "value": 425098445.5,
            "unit": "ns",
            "extra": "gctime=1814784\nmemory=363701768\nallocs=299\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/ReverseDiff/(64, 64, 16, 128)",
            "value": 494104527,
            "unit": "ns",
            "extra": "gctime=0\nmemory=235664480\nallocs=195\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Flux/(64, 64, 16, 128)",
            "value": 375779602,
            "unit": "ns",
            "extra": "gctime=0\nmemory=139122704\nallocs=155\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/SimpleChains/(64, 64, 16, 128)",
            "value": 317049031.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=31520784\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Enzyme/(64, 64, 16, 128)",
            "value": 471078717.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=107626720\nallocs=139\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/NamedTuple/(64, 64, 16, 128)",
            "value": 47539749,
            "unit": "ns",
            "extra": "gctime=0\nmemory=35922880\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/ComponentArray/(64, 64, 16, 128)",
            "value": 46905764,
            "unit": "ns",
            "extra": "gctime=0\nmemory=35922896\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/Flux/(64, 64, 16, 128)",
            "value": 56265522,
            "unit": "ns",
            "extra": "gctime=0\nmemory=67412960\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/SimpleChains/(64, 64, 16, 128)",
            "value": 28878315,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/ReverseDiff (compiled)/(2000, 128)",
            "value": 19552999,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024272\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Zygote/(2000, 128)",
            "value": 19606842,
            "unit": "ns",
            "extra": "gctime=0\nmemory=19082928\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Tracker/(2000, 128)",
            "value": 23577960,
            "unit": "ns",
            "extra": "gctime=0\nmemory=59293848\nallocs=149\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/ReverseDiff/(2000, 128)",
            "value": 24212297,
            "unit": "ns",
            "extra": "gctime=0\nmemory=39178656\nallocs=66\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Flux/(2000, 128)",
            "value": 19666577,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20105344\nallocs=33\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Enzyme/(2000, 128)",
            "value": 21011715,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2048240\nallocs=7\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/NamedTuple/(2000, 128)",
            "value": 6537708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024128\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/ComponentArray/(2000, 128)",
            "value": 6511419,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024128\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/Flux/(2000, 128)",
            "value": 6527700,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2048096\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "avik.pal.2017@gmail.com",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "distinct": true,
          "id": "6a4453d395e1433984b15d26512f1d09e3d414ea",
          "message": "test: more parallel testing based on the machine",
          "timestamp": "2024-07-25T21:27:03-07:00",
          "tree_id": "1b5468ec7b63ac316cae9cdb34f5be41b820e92b",
          "url": "https://github.com/LuxDL/Lux.jl/commit/6a4453d395e1433984b15d26512f1d09e3d414ea"
        },
        "date": 1721974727284,
        "tool": "julia",
        "benches": [
          {
            "name": "Dense(2 => 2)/cpu/reverse/ReverseDiff (compiled)/(2, 128)",
            "value": 3733.6666666666665,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":9,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Zygote/(2, 128)",
            "value": 7376.714285714285,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6016\nallocs=42\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Tracker/(2, 128)",
            "value": 20980,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17224\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/ReverseDiff/(2, 128)",
            "value": 9760,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9968\nallocs=59\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Flux/(2, 128)",
            "value": 8970.7,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5472\nallocs=28\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":5,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/SimpleChains/(2, 128)",
            "value": 4422.666666666667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3120\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":9,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Enzyme/(2, 128)",
            "value": 4676.25,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2320\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/NamedTuple/(2, 128)",
            "value": 1107.1069182389938,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1168\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":159,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/ComponentArray/(2, 128)",
            "value": 1164.4285714285713,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1168\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":147,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/Flux/(2, 128)",
            "value": 1789.4313725490197,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2176\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":51,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/SimpleChains/(2, 128)",
            "value": 179.6459802538787,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":709,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/ReverseDiff (compiled)/(20, 128)",
            "value": 17112,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10592\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Zygote/(20, 128)",
            "value": 16811,
            "unit": "ns",
            "extra": "gctime=0\nmemory=35664\nallocs=42\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Tracker/(20, 128)",
            "value": 37100,
            "unit": "ns",
            "extra": "gctime=0\nmemory=124696\nallocs=135\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/ReverseDiff/(20, 128)",
            "value": 28213,
            "unit": "ns",
            "extra": "gctime=0\nmemory=78432\nallocs=57\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Flux/(20, 128)",
            "value": 19947,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44400\nallocs=28\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/SimpleChains/(20, 128)",
            "value": 17192,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17632\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Enzyme/(20, 128)",
            "value": 25488,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20880\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/NamedTuple/(20, 128)",
            "value": 3822.25,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10448\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/ComponentArray/(20, 128)",
            "value": 3913.625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10448\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/Flux/(20, 128)",
            "value": 4784.714285714285,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20736\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/SimpleChains/(20, 128)",
            "value": 1651.1,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":10,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 3, 128)",
            "value": 39070284,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13063408\nallocs=87\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Zygote/(64, 64, 3, 128)",
            "value": 58211891,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20187840\nallocs=131\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Tracker/(64, 64, 3, 128)",
            "value": 77724823,
            "unit": "ns",
            "extra": "gctime=0\nmemory=68205704\nallocs=301\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/ReverseDiff/(64, 64, 3, 128)",
            "value": 89555817,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44197424\nallocs=197\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Flux/(64, 64, 3, 128)",
            "value": 88370701,
            "unit": "ns",
            "extra": "gctime=0\nmemory=26098864\nallocs=155\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/SimpleChains/(64, 64, 3, 128)",
            "value": 11594550,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5908032\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Enzyme/(64, 64, 3, 128)",
            "value": 91959934,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20188592\nallocs=139\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/NamedTuple/(64, 64, 3, 128)",
            "value": 7684647,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6739264\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/ComponentArray/(64, 64, 3, 128)",
            "value": 7572126.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6739280\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/Flux/(64, 64, 3, 128)",
            "value": 9887620.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12643680\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/SimpleChains/(64, 64, 3, 128)",
            "value": 6379262,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 16)",
            "value": 680001467,
            "unit": "ns",
            "extra": "gctime=0\nmemory=353327712\nallocs=4003\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 64)",
            "value": 2574834317,
            "unit": "ns",
            "extra": "gctime=52412510\nmemory=906193312\nallocs=4003\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 2)",
            "value": 133556588.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=192077392\nallocs=3969\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 16)",
            "value": 832059382,
            "unit": "ns",
            "extra": "gctime=64476402\nmemory=659570112\nallocs=10889\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 64)",
            "value": 2940015627,
            "unit": "ns",
            "extra": "gctime=324043958\nmemory=1324511808\nallocs=10888\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 2)",
            "value": 219247861,
            "unit": "ns",
            "extra": "gctime=6077679\nmemory=465628368\nallocs=10848\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 16)",
            "value": 712943058.5,
            "unit": "ns",
            "extra": "gctime=2331472\nmemory=390042496\nallocs=10658\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 64)",
            "value": 2615778063,
            "unit": "ns",
            "extra": "gctime=54111045\nmemory=1052224096\nallocs=10658\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 2)",
            "value": 129342095,
            "unit": "ns",
            "extra": "gctime=0\nmemory=196908704\nallocs=10618\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 16)",
            "value": 175664907,
            "unit": "ns",
            "extra": "gctime=0\nmemory=51373296\nallocs=977\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 64)",
            "value": 664305664.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=165112976\nallocs=977\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 2)",
            "value": 45501101,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18199712\nallocs=969\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 16)",
            "value": 165819611,
            "unit": "ns",
            "extra": "gctime=0\nmemory=51378224\nallocs=1022\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 64)",
            "value": 651918836,
            "unit": "ns",
            "extra": "gctime=0\nmemory=165117904\nallocs=1022\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 2)",
            "value": 30062079,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18204640\nallocs=1014\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 16)",
            "value": 186124645.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=69608176\nallocs=968\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 64)",
            "value": 724968189.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=238006832\nallocs=968\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 2)",
            "value": 36094692,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20492560\nallocs=957\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 64, 128)",
            "value": 1304162516.5,
            "unit": "ns",
            "extra": "gctime=169241573.5\nmemory=278646368\nallocs=88\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Zygote/(64, 64, 64, 128)",
            "value": 1868331598,
            "unit": "ns",
            "extra": "gctime=3690165.5\nmemory=430580128\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Tracker/(64, 64, 64, 128)",
            "value": 2245902418,
            "unit": "ns",
            "extra": "gctime=380865833\nmemory=1455080072\nallocs=302\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/ReverseDiff/(64, 64, 64, 128)",
            "value": 2495872396,
            "unit": "ns",
            "extra": "gctime=332531349\nmemory=942830496\nallocs=197\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Flux/(64, 64, 64, 128)",
            "value": 1950101995.5,
            "unit": "ns",
            "extra": "gctime=1764022\nmemory=556546960\nallocs=156\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Enzyme/(64, 64, 64, 128)",
            "value": 2157665806,
            "unit": "ns",
            "extra": "gctime=0\nmemory=430580640\nallocs=140\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/NamedTuple/(64, 64, 64, 128)",
            "value": 330073872,
            "unit": "ns",
            "extra": "gctime=0\nmemory=143677888\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/ComponentArray/(64, 64, 64, 128)",
            "value": 327700026,
            "unit": "ns",
            "extra": "gctime=0\nmemory=143677904\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/Flux/(64, 64, 64, 128)",
            "value": 458706730,
            "unit": "ns",
            "extra": "gctime=0\nmemory=269638112\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 1, 128)",
            "value": 11936295,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4360336\nallocs=87\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Zygote/(64, 64, 1, 128)",
            "value": 17958576,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6736912\nallocs=131\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Tracker/(64, 64, 1, 128)",
            "value": 19018839.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22747992\nallocs=299\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/ReverseDiff/(64, 64, 1, 128)",
            "value": 23737989,
            "unit": "ns",
            "extra": "gctime=0\nmemory=14743104\nallocs=195\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Flux/(64, 64, 1, 128)",
            "value": 17752292.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8711680\nallocs=155\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/SimpleChains/(64, 64, 1, 128)",
            "value": 1158352,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1970896\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Enzyme/(64, 64, 1, 128)",
            "value": 22921221,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6737680\nallocs=139\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/NamedTuple/(64, 64, 1, 128)",
            "value": 2438307,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2249472\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/ComponentArray/(64, 64, 1, 128)",
            "value": 2207289,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2249488\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/Flux/(64, 64, 1, 128)",
            "value": 2059552,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4217632\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/SimpleChains/(64, 64, 1, 128)",
            "value": 193042,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/ReverseDiff (compiled)/(200, 128)",
            "value": 290375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102672\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Zygote/(200, 128)",
            "value": 264375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=470896\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Tracker/(200, 128)",
            "value": 365440,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1614552\nallocs=149\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/ReverseDiff/(200, 128)",
            "value": 409918,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1040224\nallocs=66\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Flux/(200, 128)",
            "value": 280316,
            "unit": "ns",
            "extra": "gctime=0\nmemory=571712\nallocs=33\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/SimpleChains/(200, 128)",
            "value": 405440,
            "unit": "ns",
            "extra": "gctime=0\nmemory=747872\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Enzyme/(200, 128)",
            "value": 396112,
            "unit": "ns",
            "extra": "gctime=0\nmemory=205040\nallocs=7\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/NamedTuple/(200, 128)",
            "value": 81703,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102528\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/ComponentArray/(200, 128)",
            "value": 83426,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102528\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/Flux/(200, 128)",
            "value": 88967,
            "unit": "ns",
            "extra": "gctime=0\nmemory=204896\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/SimpleChains/(200, 128)",
            "value": 104306,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 16, 128)",
            "value": 199911623,
            "unit": "ns",
            "extra": "gctime=0\nmemory=69640608\nallocs=87\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Zygote/(64, 64, 16, 128)",
            "value": 330582216.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=107626016\nallocs=131\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Tracker/(64, 64, 16, 128)",
            "value": 431440040,
            "unit": "ns",
            "extra": "gctime=1751429\nmemory=363701768\nallocs=299\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/ReverseDiff/(64, 64, 16, 128)",
            "value": 483202494,
            "unit": "ns",
            "extra": "gctime=0\nmemory=235664480\nallocs=195\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Flux/(64, 64, 16, 128)",
            "value": 390600061,
            "unit": "ns",
            "extra": "gctime=0\nmemory=139122704\nallocs=155\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/SimpleChains/(64, 64, 16, 128)",
            "value": 329500710.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=31520784\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Enzyme/(64, 64, 16, 128)",
            "value": 476877423.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=107626720\nallocs=139\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/NamedTuple/(64, 64, 16, 128)",
            "value": 47470381,
            "unit": "ns",
            "extra": "gctime=0\nmemory=35922880\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/ComponentArray/(64, 64, 16, 128)",
            "value": 46946569,
            "unit": "ns",
            "extra": "gctime=0\nmemory=35922896\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/Flux/(64, 64, 16, 128)",
            "value": 60012110,
            "unit": "ns",
            "extra": "gctime=0\nmemory=67412960\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/SimpleChains/(64, 64, 16, 128)",
            "value": 27667373.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/ReverseDiff (compiled)/(2000, 128)",
            "value": 19236897,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024272\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Zygote/(2000, 128)",
            "value": 19653638,
            "unit": "ns",
            "extra": "gctime=0\nmemory=19082928\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Tracker/(2000, 128)",
            "value": 23866747,
            "unit": "ns",
            "extra": "gctime=0\nmemory=59293848\nallocs=149\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/ReverseDiff/(2000, 128)",
            "value": 24431304,
            "unit": "ns",
            "extra": "gctime=0\nmemory=39178656\nallocs=66\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Flux/(2000, 128)",
            "value": 19723207.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20105344\nallocs=33\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Enzyme/(2000, 128)",
            "value": 21004501,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2048240\nallocs=7\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/NamedTuple/(2000, 128)",
            "value": 6534005,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024128\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/ComponentArray/(2000, 128)",
            "value": 6514002,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024128\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/Flux/(2000, 128)",
            "value": 6501944.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2048096\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "avik.pal.2017@gmail.com",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "distinct": true,
          "id": "2a558298339020cddc481152fd8ea964e5889339",
          "message": "fix: missing module",
          "timestamp": "2024-07-26T10:25:20-07:00",
          "tree_id": "588aa2fe203462e526c313aca3abfed028f2cbbe",
          "url": "https://github.com/LuxDL/Lux.jl/commit/2a558298339020cddc481152fd8ea964e5889339"
        },
        "date": 1722016989016,
        "tool": "julia",
        "benches": [
          {
            "name": "Dense(2 => 2)/cpu/reverse/ReverseDiff (compiled)/(2, 128)",
            "value": 3661.875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Zygote/(2, 128)",
            "value": 7419.8,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6016\nallocs=42\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":5,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Tracker/(2, 128)",
            "value": 20999,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17224\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/ReverseDiff/(2, 128)",
            "value": 9738,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9968\nallocs=59\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Flux/(2, 128)",
            "value": 9050.8,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5472\nallocs=28\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":5,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/SimpleChains/(2, 128)",
            "value": 4475.875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3120\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Enzyme/(2, 128)",
            "value": 4693.875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2320\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/NamedTuple/(2, 128)",
            "value": 1112.656050955414,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1168\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":157,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/ComponentArray/(2, 128)",
            "value": 1169.8661971830986,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1168\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":142,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/Flux/(2, 128)",
            "value": 1777.3272727272727,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2176\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":55,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/SimpleChains/(2, 128)",
            "value": 179.5702364394993,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":719,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/ReverseDiff (compiled)/(20, 128)",
            "value": 17302,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10592\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Zygote/(20, 128)",
            "value": 17032,
            "unit": "ns",
            "extra": "gctime=0\nmemory=35664\nallocs=42\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Tracker/(20, 128)",
            "value": 37040,
            "unit": "ns",
            "extra": "gctime=0\nmemory=124696\nallocs=135\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/ReverseDiff/(20, 128)",
            "value": 29184,
            "unit": "ns",
            "extra": "gctime=0\nmemory=78432\nallocs=57\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Flux/(20, 128)",
            "value": 21551,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44400\nallocs=28\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/SimpleChains/(20, 128)",
            "value": 17293,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17632\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Enzyme/(20, 128)",
            "value": 25498,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20880\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/NamedTuple/(20, 128)",
            "value": 3844.75,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10448\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/ComponentArray/(20, 128)",
            "value": 3933.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10448\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/Flux/(20, 128)",
            "value": 4952.142857142857,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20736\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/SimpleChains/(20, 128)",
            "value": 1654.1,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":10,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 3, 128)",
            "value": 50721668.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13063408\nallocs=87\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Zygote/(64, 64, 3, 128)",
            "value": 58546553,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20187840\nallocs=131\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Tracker/(64, 64, 3, 128)",
            "value": 101678849,
            "unit": "ns",
            "extra": "gctime=0\nmemory=68205704\nallocs=301\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/ReverseDiff/(64, 64, 3, 128)",
            "value": 101190618,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44197424\nallocs=197\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Flux/(64, 64, 3, 128)",
            "value": 78796997,
            "unit": "ns",
            "extra": "gctime=0\nmemory=26098864\nallocs=155\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/SimpleChains/(64, 64, 3, 128)",
            "value": 12343077,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5908032\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Enzyme/(64, 64, 3, 128)",
            "value": 92604699,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20188592\nallocs=139\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/NamedTuple/(64, 64, 3, 128)",
            "value": 7675430,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6739264\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/ComponentArray/(64, 64, 3, 128)",
            "value": 7608555,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6739280\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/Flux/(64, 64, 3, 128)",
            "value": 12512726,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12643680\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/SimpleChains/(64, 64, 3, 128)",
            "value": 6420844,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 16)",
            "value": 698316993,
            "unit": "ns",
            "extra": "gctime=0\nmemory=353327712\nallocs=4003\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 64)",
            "value": 2576787594,
            "unit": "ns",
            "extra": "gctime=52934854\nmemory=906193312\nallocs=4003\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 2)",
            "value": 141986062.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=192077392\nallocs=3969\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 16)",
            "value": 893749885.5,
            "unit": "ns",
            "extra": "gctime=31313347\nmemory=659567936\nallocs=10888\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 64)",
            "value": 3362911269,
            "unit": "ns",
            "extra": "gctime=391676069\nmemory=1324513984\nallocs=10889\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 2)",
            "value": 207659968,
            "unit": "ns",
            "extra": "gctime=1773112\nmemory=465628368\nallocs=10848\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 16)",
            "value": 857934198,
            "unit": "ns",
            "extra": "gctime=2996547\nmemory=390042496\nallocs=10658\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 64)",
            "value": 2864984030,
            "unit": "ns",
            "extra": "gctime=54607017\nmemory=1052224096\nallocs=10658\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 2)",
            "value": 149742910,
            "unit": "ns",
            "extra": "gctime=0\nmemory=196908704\nallocs=10618\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 16)",
            "value": 177136415.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=51373296\nallocs=977\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 64)",
            "value": 662230879,
            "unit": "ns",
            "extra": "gctime=0\nmemory=165112976\nallocs=977\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 2)",
            "value": 36459877,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18199712\nallocs=969\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 16)",
            "value": 167990874.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=51378224\nallocs=1022\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 64)",
            "value": 655057499,
            "unit": "ns",
            "extra": "gctime=0\nmemory=165117904\nallocs=1022\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 2)",
            "value": 35749851,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18204640\nallocs=1014\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 16)",
            "value": 228962990.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=69608176\nallocs=968\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 64)",
            "value": 847928462,
            "unit": "ns",
            "extra": "gctime=0\nmemory=238006832\nallocs=968\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 2)",
            "value": 37808714,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20492560\nallocs=957\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 64, 128)",
            "value": 1328888452,
            "unit": "ns",
            "extra": "gctime=176681133\nmemory=278646368\nallocs=88\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Zygote/(64, 64, 64, 128)",
            "value": 1883995892,
            "unit": "ns",
            "extra": "gctime=3541954\nmemory=430580128\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Tracker/(64, 64, 64, 128)",
            "value": 2418351457,
            "unit": "ns",
            "extra": "gctime=394124584\nmemory=1455080072\nallocs=302\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/ReverseDiff/(64, 64, 64, 128)",
            "value": 2440238931,
            "unit": "ns",
            "extra": "gctime=344709028\nmemory=942830496\nallocs=197\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Flux/(64, 64, 64, 128)",
            "value": 1948384572,
            "unit": "ns",
            "extra": "gctime=1422716.5\nmemory=556546960\nallocs=156\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Enzyme/(64, 64, 64, 128)",
            "value": 2084376262,
            "unit": "ns",
            "extra": "gctime=0\nmemory=430580640\nallocs=140\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/NamedTuple/(64, 64, 64, 128)",
            "value": 336503625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=143677888\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/ComponentArray/(64, 64, 64, 128)",
            "value": 336582406,
            "unit": "ns",
            "extra": "gctime=0\nmemory=143677904\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/Flux/(64, 64, 64, 128)",
            "value": 456801176,
            "unit": "ns",
            "extra": "gctime=0\nmemory=269638112\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 1, 128)",
            "value": 11858627,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4360336\nallocs=87\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Zygote/(64, 64, 1, 128)",
            "value": 18114833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6736912\nallocs=131\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Tracker/(64, 64, 1, 128)",
            "value": 19305583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22747992\nallocs=299\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/ReverseDiff/(64, 64, 1, 128)",
            "value": 24143369.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=14743104\nallocs=195\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Flux/(64, 64, 1, 128)",
            "value": 17990331,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8711680\nallocs=155\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/SimpleChains/(64, 64, 1, 128)",
            "value": 1181481.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1970896\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Enzyme/(64, 64, 1, 128)",
            "value": 23281651.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6737680\nallocs=139\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/NamedTuple/(64, 64, 1, 128)",
            "value": 2329001,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2249472\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/ComponentArray/(64, 64, 1, 128)",
            "value": 2259670,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2249488\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/Flux/(64, 64, 1, 128)",
            "value": 2106879,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4217632\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/SimpleChains/(64, 64, 1, 128)",
            "value": 216514,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/ReverseDiff (compiled)/(200, 128)",
            "value": 295862,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102672\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Zygote/(200, 128)",
            "value": 269132.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=470896\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Tracker/(200, 128)",
            "value": 370853,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1614552\nallocs=149\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/ReverseDiff/(200, 128)",
            "value": 416127,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1040224\nallocs=66\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Flux/(200, 128)",
            "value": 278089,
            "unit": "ns",
            "extra": "gctime=0\nmemory=571712\nallocs=33\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/SimpleChains/(200, 128)",
            "value": 409124,
            "unit": "ns",
            "extra": "gctime=0\nmemory=747872\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Enzyme/(200, 128)",
            "value": 399266,
            "unit": "ns",
            "extra": "gctime=0\nmemory=205040\nallocs=7\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/NamedTuple/(200, 128)",
            "value": 84137,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102528\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/ComponentArray/(200, 128)",
            "value": 86591,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102528\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/Flux/(200, 128)",
            "value": 87734,
            "unit": "ns",
            "extra": "gctime=0\nmemory=204896\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/SimpleChains/(200, 128)",
            "value": 104285,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 16, 128)",
            "value": 196679056,
            "unit": "ns",
            "extra": "gctime=0\nmemory=69640608\nallocs=87\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Zygote/(64, 64, 16, 128)",
            "value": 331090526,
            "unit": "ns",
            "extra": "gctime=0\nmemory=107626016\nallocs=131\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Tracker/(64, 64, 16, 128)",
            "value": 446330080.5,
            "unit": "ns",
            "extra": "gctime=1765281.5\nmemory=363701768\nallocs=299\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/ReverseDiff/(64, 64, 16, 128)",
            "value": 501425011,
            "unit": "ns",
            "extra": "gctime=0\nmemory=235664480\nallocs=195\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Flux/(64, 64, 16, 128)",
            "value": 419437580.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=139122704\nallocs=155\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/SimpleChains/(64, 64, 16, 128)",
            "value": 346091233,
            "unit": "ns",
            "extra": "gctime=0\nmemory=31520784\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Enzyme/(64, 64, 16, 128)",
            "value": 483112933.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=107626720\nallocs=139\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/NamedTuple/(64, 64, 16, 128)",
            "value": 47755090,
            "unit": "ns",
            "extra": "gctime=0\nmemory=35922880\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/ComponentArray/(64, 64, 16, 128)",
            "value": 47275593,
            "unit": "ns",
            "extra": "gctime=0\nmemory=35922896\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/Flux/(64, 64, 16, 128)",
            "value": 57880765,
            "unit": "ns",
            "extra": "gctime=0\nmemory=67412960\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/SimpleChains/(64, 64, 16, 128)",
            "value": 28154162.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/ReverseDiff (compiled)/(2000, 128)",
            "value": 19633406,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024272\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Zygote/(2000, 128)",
            "value": 19863886,
            "unit": "ns",
            "extra": "gctime=0\nmemory=19082928\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Tracker/(2000, 128)",
            "value": 23978995,
            "unit": "ns",
            "extra": "gctime=0\nmemory=59293848\nallocs=149\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/ReverseDiff/(2000, 128)",
            "value": 24483866,
            "unit": "ns",
            "extra": "gctime=0\nmemory=39178656\nallocs=66\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Flux/(2000, 128)",
            "value": 19853090.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20105344\nallocs=33\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Enzyme/(2000, 128)",
            "value": 21211617,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2048240\nallocs=7\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/NamedTuple/(2000, 128)",
            "value": 6587852,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024128\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/ComponentArray/(2000, 128)",
            "value": 6555306,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024128\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/Flux/(2000, 128)",
            "value": 6543112,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2048096\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "distinct": true,
          "id": "f3159bda9f1244fa6e5fe54cd6e7b4c2a3a53dc9",
          "message": "test: remove redundant timeout",
          "timestamp": "2024-07-27T15:43:07-07:00",
          "tree_id": "6ae7161deff9c0ef1ea0d95de00503bdfb236d17",
          "url": "https://github.com/LuxDL/Lux.jl/commit/f3159bda9f1244fa6e5fe54cd6e7b4c2a3a53dc9"
        },
        "date": 1722122521467,
        "tool": "julia",
        "benches": [
          {
            "name": "Dense(2 => 2)/cpu/reverse/ReverseDiff (compiled)/(2, 128)",
            "value": 4340.625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Zygote/(2, 128)",
            "value": 8991.833333333334,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6016\nallocs=42\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Tracker/(2, 128)",
            "value": 25788,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17224\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/ReverseDiff/(2, 128)",
            "value": 12129.333333333334,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9968\nallocs=59\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Flux/(2, 128)",
            "value": 10748.1,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5472\nallocs=28\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":5,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/SimpleChains/(2, 128)",
            "value": 4954.25,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3120\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Enzyme/(2, 128)",
            "value": 5425.125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2320\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/NamedTuple/(2, 128)",
            "value": 1359.1832061068703,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1168\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":131,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/ComponentArray/(2, 128)",
            "value": 1401.2664233576643,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1168\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":137,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/Flux/(2, 128)",
            "value": 2176.5833333333335,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2176\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":48,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/SimpleChains/(2, 128)",
            "value": 179.46778711484595,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":714,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/ReverseDiff (compiled)/(20, 128)",
            "value": 21761,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10592\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Zygote/(20, 128)",
            "value": 20408,
            "unit": "ns",
            "extra": "gctime=0\nmemory=35664\nallocs=42\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Tracker/(20, 128)",
            "value": 42780,
            "unit": "ns",
            "extra": "gctime=0\nmemory=124696\nallocs=135\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/ReverseDiff/(20, 128)",
            "value": 32301,
            "unit": "ns",
            "extra": "gctime=0\nmemory=78432\nallocs=57\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Flux/(20, 128)",
            "value": 22652,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44400\nallocs=28\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/SimpleChains/(20, 128)",
            "value": 18654,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17632\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Enzyme/(20, 128)",
            "value": 28283,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20880\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/NamedTuple/(20, 128)",
            "value": 4363.8125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10448\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/ComponentArray/(20, 128)",
            "value": 4459.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10448\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/Flux/(20, 128)",
            "value": 5581.928571428572,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20736\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/SimpleChains/(20, 128)",
            "value": 1850.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":10,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 3, 128)",
            "value": 45154820.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13063408\nallocs=87\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Zygote/(64, 64, 3, 128)",
            "value": 62378075.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20187840\nallocs=131\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Tracker/(64, 64, 3, 128)",
            "value": 82237576.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=68205704\nallocs=301\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/ReverseDiff/(64, 64, 3, 128)",
            "value": 92405677.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44197424\nallocs=197\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Flux/(64, 64, 3, 128)",
            "value": 78467695,
            "unit": "ns",
            "extra": "gctime=0\nmemory=26098864\nallocs=155\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/SimpleChains/(64, 64, 3, 128)",
            "value": 11963195.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5908032\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Enzyme/(64, 64, 3, 128)",
            "value": 91650799,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20188592\nallocs=139\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/NamedTuple/(64, 64, 3, 128)",
            "value": 7710446.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6739264\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/ComponentArray/(64, 64, 3, 128)",
            "value": 7584631,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6739280\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/Flux/(64, 64, 3, 128)",
            "value": 12087804.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12643680\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/SimpleChains/(64, 64, 3, 128)",
            "value": 6389570.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 16)",
            "value": 691470864.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=317744848\nallocs=3522\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 64)",
            "value": 2612057812,
            "unit": "ns",
            "extra": "gctime=58060707\nmemory=764442128\nallocs=3522\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 2)",
            "value": 151710246,
            "unit": "ns",
            "extra": "gctime=0\nmemory=187459904\nallocs=3494\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 16)",
            "value": 847276341,
            "unit": "ns",
            "extra": "gctime=55753016\nmemory=659567936\nallocs=10888\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 64)",
            "value": 2998224779,
            "unit": "ns",
            "extra": "gctime=324029960\nmemory=1324511808\nallocs=10888\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 2)",
            "value": 231162547,
            "unit": "ns",
            "extra": "gctime=27420568\nmemory=465628368\nallocs=10848\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 16)",
            "value": 712811787.5,
            "unit": "ns",
            "extra": "gctime=3697569\nmemory=390042496\nallocs=10658\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 64)",
            "value": 2949177680,
            "unit": "ns",
            "extra": "gctime=52744290\nmemory=1052224096\nallocs=10658\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 2)",
            "value": 139771741,
            "unit": "ns",
            "extra": "gctime=0\nmemory=196908704\nallocs=10618\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 16)",
            "value": 175403229,
            "unit": "ns",
            "extra": "gctime=0\nmemory=51375328\nallocs=1042\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 64)",
            "value": 643986392.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=165115008\nallocs=1042\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 2)",
            "value": 45299778,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18201744\nallocs=1034\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 16)",
            "value": 165572824,
            "unit": "ns",
            "extra": "gctime=0\nmemory=51381088\nallocs=1087\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 64)",
            "value": 648012066,
            "unit": "ns",
            "extra": "gctime=0\nmemory=165120768\nallocs=1087\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 2)",
            "value": 35881786,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18207504\nallocs=1079\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 16)",
            "value": 228284686,
            "unit": "ns",
            "extra": "gctime=0\nmemory=69608176\nallocs=968\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 64)",
            "value": 824337722,
            "unit": "ns",
            "extra": "gctime=0\nmemory=238006832\nallocs=968\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 2)",
            "value": 36049946,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20492560\nallocs=957\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 64, 128)",
            "value": 1305860557.5,
            "unit": "ns",
            "extra": "gctime=172100719\nmemory=278646368\nallocs=88\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Zygote/(64, 64, 64, 128)",
            "value": 1872497449,
            "unit": "ns",
            "extra": "gctime=3940933.5\nmemory=430580128\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Tracker/(64, 64, 64, 128)",
            "value": 2342556041,
            "unit": "ns",
            "extra": "gctime=373865602\nmemory=1455080072\nallocs=302\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/ReverseDiff/(64, 64, 64, 128)",
            "value": 2488038610,
            "unit": "ns",
            "extra": "gctime=338907534\nmemory=942830496\nallocs=197\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Flux/(64, 64, 64, 128)",
            "value": 1951631856,
            "unit": "ns",
            "extra": "gctime=2411010\nmemory=556546960\nallocs=156\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Enzyme/(64, 64, 64, 128)",
            "value": 2087636119,
            "unit": "ns",
            "extra": "gctime=0\nmemory=430580640\nallocs=140\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/NamedTuple/(64, 64, 64, 128)",
            "value": 338033674,
            "unit": "ns",
            "extra": "gctime=0\nmemory=143677888\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/ComponentArray/(64, 64, 64, 128)",
            "value": 332696625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=143677904\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/Flux/(64, 64, 64, 128)",
            "value": 359960422,
            "unit": "ns",
            "extra": "gctime=0\nmemory=269638112\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 1, 128)",
            "value": 11686611,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4360336\nallocs=87\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Zygote/(64, 64, 1, 128)",
            "value": 17986098.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6736912\nallocs=131\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Tracker/(64, 64, 1, 128)",
            "value": 19052140,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22747992\nallocs=299\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/ReverseDiff/(64, 64, 1, 128)",
            "value": 23778464,
            "unit": "ns",
            "extra": "gctime=0\nmemory=14743104\nallocs=195\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Flux/(64, 64, 1, 128)",
            "value": 17775034.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8711680\nallocs=155\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/SimpleChains/(64, 64, 1, 128)",
            "value": 1157462,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1970896\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Enzyme/(64, 64, 1, 128)",
            "value": 22924969.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6737680\nallocs=139\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/NamedTuple/(64, 64, 1, 128)",
            "value": 2406465,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2249472\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/ComponentArray/(64, 64, 1, 128)",
            "value": 2242648.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2249488\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/Flux/(64, 64, 1, 128)",
            "value": 2065648,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4217632\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/SimpleChains/(64, 64, 1, 128)",
            "value": 197910,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/ReverseDiff (compiled)/(200, 128)",
            "value": 290874,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102672\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Zygote/(200, 128)",
            "value": 264424,
            "unit": "ns",
            "extra": "gctime=0\nmemory=470896\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Tracker/(200, 128)",
            "value": 364661,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1614552\nallocs=149\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/ReverseDiff/(200, 128)",
            "value": 405888,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1040224\nallocs=66\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Flux/(200, 128)",
            "value": 272550,
            "unit": "ns",
            "extra": "gctime=0\nmemory=571712\nallocs=33\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/SimpleChains/(200, 128)",
            "value": 408042,
            "unit": "ns",
            "extra": "gctime=0\nmemory=747872\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Enzyme/(200, 128)",
            "value": 394657,
            "unit": "ns",
            "extra": "gctime=0\nmemory=205040\nallocs=7\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/NamedTuple/(200, 128)",
            "value": 80941,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102528\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/ComponentArray/(200, 128)",
            "value": 81403,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102528\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/Flux/(200, 128)",
            "value": 86461,
            "unit": "ns",
            "extra": "gctime=0\nmemory=204896\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/SimpleChains/(200, 128)",
            "value": 104525,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 16, 128)",
            "value": 206243721,
            "unit": "ns",
            "extra": "gctime=0\nmemory=69640608\nallocs=87\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Zygote/(64, 64, 16, 128)",
            "value": 327524051.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=107626016\nallocs=131\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Tracker/(64, 64, 16, 128)",
            "value": 410738774,
            "unit": "ns",
            "extra": "gctime=1702911\nmemory=363701768\nallocs=299\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/ReverseDiff/(64, 64, 16, 128)",
            "value": 439152524.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=235664480\nallocs=195\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Flux/(64, 64, 16, 128)",
            "value": 382717413,
            "unit": "ns",
            "extra": "gctime=0\nmemory=139122704\nallocs=155\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/SimpleChains/(64, 64, 16, 128)",
            "value": 325870578.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=31520784\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Enzyme/(64, 64, 16, 128)",
            "value": 455578258,
            "unit": "ns",
            "extra": "gctime=0\nmemory=107626720\nallocs=139\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/NamedTuple/(64, 64, 16, 128)",
            "value": 47303828.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=35922880\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/ComponentArray/(64, 64, 16, 128)",
            "value": 46793042.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=35922896\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/Flux/(64, 64, 16, 128)",
            "value": 57476949,
            "unit": "ns",
            "extra": "gctime=0\nmemory=67412960\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/SimpleChains/(64, 64, 16, 128)",
            "value": 27955898,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/ReverseDiff (compiled)/(2000, 128)",
            "value": 18906265,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024272\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Zygote/(2000, 128)",
            "value": 19556819,
            "unit": "ns",
            "extra": "gctime=0\nmemory=19082928\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Tracker/(2000, 128)",
            "value": 23268022,
            "unit": "ns",
            "extra": "gctime=0\nmemory=59293848\nallocs=149\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/ReverseDiff/(2000, 128)",
            "value": 24118160,
            "unit": "ns",
            "extra": "gctime=0\nmemory=39178656\nallocs=66\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Flux/(2000, 128)",
            "value": 19557901,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20105344\nallocs=33\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Enzyme/(2000, 128)",
            "value": 20840171,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2048240\nallocs=7\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/NamedTuple/(2000, 128)",
            "value": 6539910.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024128\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/ComponentArray/(2000, 128)",
            "value": 6488719,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024128\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/Flux/(2000, 128)",
            "value": 6511897,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2048096\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "avik.pal.2017@gmail.com",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "distinct": true,
          "id": "8eb392aa6104c845a45593b663b00d9bce61b107",
          "message": "docs: update broken links and fail on linkcheck",
          "timestamp": "2024-07-27T22:15:33-07:00",
          "tree_id": "bf9090e0321762d503b4dc99a6b5cac80409586c",
          "url": "https://github.com/LuxDL/Lux.jl/commit/8eb392aa6104c845a45593b663b00d9bce61b107"
        },
        "date": 1722147807195,
        "tool": "julia",
        "benches": [
          {
            "name": "Dense(2 => 2)/cpu/reverse/ReverseDiff (compiled)/(2, 128)",
            "value": 3679.375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Zygote/(2, 128)",
            "value": 7338,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6016\nallocs=42\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Tracker/(2, 128)",
            "value": 20558,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17224\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/ReverseDiff/(2, 128)",
            "value": 9722.2,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9968\nallocs=59\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":5,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Flux/(2, 128)",
            "value": 9037,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5472\nallocs=28\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":4,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/SimpleChains/(2, 128)",
            "value": 4463.9375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3120\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Enzyme/(2, 128)",
            "value": 4666.25,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2320\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/NamedTuple/(2, 128)",
            "value": 1114.3974358974358,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1168\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":156,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/ComponentArray/(2, 128)",
            "value": 1184.1407407407407,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1168\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":135,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/Flux/(2, 128)",
            "value": 1810.445652173913,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2176\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":46,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/SimpleChains/(2, 128)",
            "value": 180.16946778711485,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":714,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/ReverseDiff (compiled)/(20, 128)",
            "value": 17292,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10592\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Zygote/(20, 128)",
            "value": 16852,
            "unit": "ns",
            "extra": "gctime=0\nmemory=35664\nallocs=42\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Tracker/(20, 128)",
            "value": 36358,
            "unit": "ns",
            "extra": "gctime=0\nmemory=124696\nallocs=135\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/ReverseDiff/(20, 128)",
            "value": 28133,
            "unit": "ns",
            "extra": "gctime=0\nmemory=78432\nallocs=57\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Flux/(20, 128)",
            "value": 20148,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44400\nallocs=28\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/SimpleChains/(20, 128)",
            "value": 17052,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17632\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Enzyme/(20, 128)",
            "value": 25548,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20880\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/NamedTuple/(20, 128)",
            "value": 3795.875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10448\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/ComponentArray/(20, 128)",
            "value": 3908.625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10448\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/Flux/(20, 128)",
            "value": 4793.285714285715,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20736\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/SimpleChains/(20, 128)",
            "value": 1655.1,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":10,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 3, 128)",
            "value": 38607620.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13063408\nallocs=87\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Zygote/(64, 64, 3, 128)",
            "value": 58062464,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20187840\nallocs=131\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Tracker/(64, 64, 3, 128)",
            "value": 66821661,
            "unit": "ns",
            "extra": "gctime=0\nmemory=68205704\nallocs=301\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/ReverseDiff/(64, 64, 3, 128)",
            "value": 79893843,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44197424\nallocs=197\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Flux/(64, 64, 3, 128)",
            "value": 72220779,
            "unit": "ns",
            "extra": "gctime=0\nmemory=26098864\nallocs=155\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/SimpleChains/(64, 64, 3, 128)",
            "value": 11898610,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5908032\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Enzyme/(64, 64, 3, 128)",
            "value": 79046441.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20188592\nallocs=139\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/NamedTuple/(64, 64, 3, 128)",
            "value": 7662156.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6739264\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/ComponentArray/(64, 64, 3, 128)",
            "value": 7540729,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6739280\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/Flux/(64, 64, 3, 128)",
            "value": 10363028,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12643680\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/SimpleChains/(64, 64, 3, 128)",
            "value": 6374425,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 16)",
            "value": 688891366,
            "unit": "ns",
            "extra": "gctime=0\nmemory=317744848\nallocs=3522\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 64)",
            "value": 2580089696,
            "unit": "ns",
            "extra": "gctime=52750651\nmemory=764442128\nallocs=3522\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 2)",
            "value": 140409324.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=187459904\nallocs=3494\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 16)",
            "value": 767316784,
            "unit": "ns",
            "extra": "gctime=54844818\nmemory=659567936\nallocs=10888\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 64)",
            "value": 2911634396,
            "unit": "ns",
            "extra": "gctime=310953365\nmemory=1324513984\nallocs=10889\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 2)",
            "value": 220564407,
            "unit": "ns",
            "extra": "gctime=3156359\nmemory=465628368\nallocs=10848\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 16)",
            "value": 660516925.5,
            "unit": "ns",
            "extra": "gctime=2343712\nmemory=390042496\nallocs=10658\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 64)",
            "value": 2628665253,
            "unit": "ns",
            "extra": "gctime=215501240\nmemory=1052224096\nallocs=10658\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 2)",
            "value": 134431743,
            "unit": "ns",
            "extra": "gctime=0\nmemory=196908704\nallocs=10618\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 16)",
            "value": 172684244,
            "unit": "ns",
            "extra": "gctime=0\nmemory=51375328\nallocs=1042\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 64)",
            "value": 660049856.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=165115008\nallocs=1042\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 2)",
            "value": 34254472,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18201744\nallocs=1034\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 16)",
            "value": 163069879,
            "unit": "ns",
            "extra": "gctime=0\nmemory=51381088\nallocs=1087\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 64)",
            "value": 639471793,
            "unit": "ns",
            "extra": "gctime=0\nmemory=165120768\nallocs=1087\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 2)",
            "value": 29383379,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18207504\nallocs=1079\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 16)",
            "value": 201738097,
            "unit": "ns",
            "extra": "gctime=0\nmemory=69608176\nallocs=968\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 64)",
            "value": 722144211.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=238006832\nallocs=968\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 2)",
            "value": 35370961,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20492560\nallocs=957\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 64, 128)",
            "value": 1247821765,
            "unit": "ns",
            "extra": "gctime=161940772.5\nmemory=278646368\nallocs=88\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Zygote/(64, 64, 64, 128)",
            "value": 1872950778,
            "unit": "ns",
            "extra": "gctime=3463544\nmemory=430580128\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Tracker/(64, 64, 64, 128)",
            "value": 2218972759.5,
            "unit": "ns",
            "extra": "gctime=344480433\nmemory=1455080072\nallocs=302\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/ReverseDiff/(64, 64, 64, 128)",
            "value": 2061757737.5,
            "unit": "ns",
            "extra": "gctime=3689196.5\nmemory=942830496\nallocs=197\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Flux/(64, 64, 64, 128)",
            "value": 1829486370,
            "unit": "ns",
            "extra": "gctime=3207565\nmemory=556546960\nallocs=156\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Enzyme/(64, 64, 64, 128)",
            "value": 2001768856,
            "unit": "ns",
            "extra": "gctime=4221763\nmemory=430580640\nallocs=140\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/NamedTuple/(64, 64, 64, 128)",
            "value": 329179369,
            "unit": "ns",
            "extra": "gctime=0\nmemory=143677888\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/ComponentArray/(64, 64, 64, 128)",
            "value": 328423712,
            "unit": "ns",
            "extra": "gctime=0\nmemory=143677904\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/Flux/(64, 64, 64, 128)",
            "value": 345125201,
            "unit": "ns",
            "extra": "gctime=0\nmemory=269638112\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 1, 128)",
            "value": 11790993.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4360336\nallocs=87\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Zygote/(64, 64, 1, 128)",
            "value": 18100299,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6736912\nallocs=131\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Tracker/(64, 64, 1, 128)",
            "value": 18996442,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22747992\nallocs=299\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/ReverseDiff/(64, 64, 1, 128)",
            "value": 23733608,
            "unit": "ns",
            "extra": "gctime=0\nmemory=14743104\nallocs=195\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Flux/(64, 64, 1, 128)",
            "value": 17885808.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8711680\nallocs=155\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/SimpleChains/(64, 64, 1, 128)",
            "value": 1164034,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1970896\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Enzyme/(64, 64, 1, 128)",
            "value": 22919399,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6737680\nallocs=139\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/NamedTuple/(64, 64, 1, 128)",
            "value": 2345181,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2249472\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/ComponentArray/(64, 64, 1, 128)",
            "value": 2211761,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2249488\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/Flux/(64, 64, 1, 128)",
            "value": 2058975,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4217632\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/SimpleChains/(64, 64, 1, 128)",
            "value": 195266,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/ReverseDiff (compiled)/(200, 128)",
            "value": 291195,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102672\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Zygote/(200, 128)",
            "value": 263513,
            "unit": "ns",
            "extra": "gctime=0\nmemory=470896\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Tracker/(200, 128)",
            "value": 358852,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1614552\nallocs=149\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/ReverseDiff/(200, 128)",
            "value": 404647,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1040224\nallocs=66\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Flux/(200, 128)",
            "value": 271508,
            "unit": "ns",
            "extra": "gctime=0\nmemory=571712\nallocs=33\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/SimpleChains/(200, 128)",
            "value": 405448,
            "unit": "ns",
            "extra": "gctime=0\nmemory=747872\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Enzyme/(200, 128)",
            "value": 394609,
            "unit": "ns",
            "extra": "gctime=0\nmemory=205040\nallocs=7\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/NamedTuple/(200, 128)",
            "value": 80831,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102528\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/ComponentArray/(200, 128)",
            "value": 81954,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102528\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/Flux/(200, 128)",
            "value": 86211,
            "unit": "ns",
            "extra": "gctime=0\nmemory=204896\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/SimpleChains/(200, 128)",
            "value": 104485,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 16, 128)",
            "value": 187479900,
            "unit": "ns",
            "extra": "gctime=0\nmemory=69640608\nallocs=87\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Zygote/(64, 64, 16, 128)",
            "value": 328956491,
            "unit": "ns",
            "extra": "gctime=0\nmemory=107626016\nallocs=131\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Tracker/(64, 64, 16, 128)",
            "value": 419428117,
            "unit": "ns",
            "extra": "gctime=1897452\nmemory=363701768\nallocs=299\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/ReverseDiff/(64, 64, 16, 128)",
            "value": 479472814.5,
            "unit": "ns",
            "extra": "gctime=768859.5\nmemory=235664480\nallocs=195\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Flux/(64, 64, 16, 128)",
            "value": 380882756,
            "unit": "ns",
            "extra": "gctime=0\nmemory=139122704\nallocs=155\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/SimpleChains/(64, 64, 16, 128)",
            "value": 306226546,
            "unit": "ns",
            "extra": "gctime=0\nmemory=31520784\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Enzyme/(64, 64, 16, 128)",
            "value": 468205037,
            "unit": "ns",
            "extra": "gctime=0\nmemory=107626720\nallocs=139\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/NamedTuple/(64, 64, 16, 128)",
            "value": 47148555,
            "unit": "ns",
            "extra": "gctime=0\nmemory=35922880\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/ComponentArray/(64, 64, 16, 128)",
            "value": 46601077.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=35922896\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/Flux/(64, 64, 16, 128)",
            "value": 59226820,
            "unit": "ns",
            "extra": "gctime=0\nmemory=67412960\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/SimpleChains/(64, 64, 16, 128)",
            "value": 28155899,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/ReverseDiff (compiled)/(2000, 128)",
            "value": 18845148.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024272\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Zygote/(2000, 128)",
            "value": 19457185,
            "unit": "ns",
            "extra": "gctime=0\nmemory=19082928\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Tracker/(2000, 128)",
            "value": 23071936.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=59293848\nallocs=149\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/ReverseDiff/(2000, 128)",
            "value": 23991051.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=39178656\nallocs=66\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Flux/(2000, 128)",
            "value": 19527991.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20105344\nallocs=33\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Enzyme/(2000, 128)",
            "value": 20815286.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2048240\nallocs=7\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/NamedTuple/(2000, 128)",
            "value": 6510728,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024128\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/ComponentArray/(2000, 128)",
            "value": 6483627,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024128\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/Flux/(2000, 128)",
            "value": 6472777,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2048096\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "avik.pal.2017@gmail.com",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "distinct": true,
          "id": "49fcfea134eda1fb7a19eb9754e069e5189c8ecc",
          "message": "fix: skip more enzyme tests and remove finitedifferences",
          "timestamp": "2024-07-28T23:20:57-07:00",
          "tree_id": "3c8f55de32469d097518099ae0fb3a1045a7806d",
          "url": "https://github.com/LuxDL/Lux.jl/commit/49fcfea134eda1fb7a19eb9754e069e5189c8ecc"
        },
        "date": 1722237407447,
        "tool": "julia",
        "benches": [
          {
            "name": "Dense(2 => 2)/cpu/reverse/ReverseDiff (compiled)/(2, 128)",
            "value": 3639.25,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Zygote/(2, 128)",
            "value": 7427.166666666667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6016\nallocs=42\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Tracker/(2, 128)",
            "value": 21300,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17224\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/ReverseDiff/(2, 128)",
            "value": 9991.25,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9968\nallocs=59\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":4,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Flux/(2, 128)",
            "value": 9222.25,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5472\nallocs=28\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":4,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/SimpleChains/(2, 128)",
            "value": 4509.625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3120\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Enzyme/(2, 128)",
            "value": 4725.6875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2320\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/NamedTuple/(2, 128)",
            "value": 1121.59375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1168\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":160,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/ComponentArray/(2, 128)",
            "value": 1200.0229007633588,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1168\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":131,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/Flux/(2, 128)",
            "value": 1793.5172413793102,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2176\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":58,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/SimpleChains/(2, 128)",
            "value": 179.46498599439775,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":714,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/ReverseDiff (compiled)/(20, 128)",
            "value": 17433,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10592\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Zygote/(20, 128)",
            "value": 16962,
            "unit": "ns",
            "extra": "gctime=0\nmemory=35664\nallocs=42\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Tracker/(20, 128)",
            "value": 37971,
            "unit": "ns",
            "extra": "gctime=0\nmemory=124696\nallocs=135\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/ReverseDiff/(20, 128)",
            "value": 29279.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=78432\nallocs=57\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Flux/(20, 128)",
            "value": 21415,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44400\nallocs=28\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/SimpleChains/(20, 128)",
            "value": 17392,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17632\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Enzyme/(20, 128)",
            "value": 25588,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20880\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/NamedTuple/(20, 128)",
            "value": 3886,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10448\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/ComponentArray/(20, 128)",
            "value": 3946.125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10448\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/Flux/(20, 128)",
            "value": 4957.857142857143,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20736\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/SimpleChains/(20, 128)",
            "value": 1652.1,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":10,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 3, 128)",
            "value": 39460360.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13063408\nallocs=87\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Zygote/(64, 64, 3, 128)",
            "value": 58691860,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20187840\nallocs=131\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Tracker/(64, 64, 3, 128)",
            "value": 77996827.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=68205704\nallocs=301\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/ReverseDiff/(64, 64, 3, 128)",
            "value": 89822634.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44197424\nallocs=197\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Flux/(64, 64, 3, 128)",
            "value": 72257808.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=26098864\nallocs=155\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/SimpleChains/(64, 64, 3, 128)",
            "value": 11941715.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5908032\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Enzyme/(64, 64, 3, 128)",
            "value": 87378870,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20188592\nallocs=139\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/NamedTuple/(64, 64, 3, 128)",
            "value": 7676034,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6739264\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/ComponentArray/(64, 64, 3, 128)",
            "value": 7626962.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6739280\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/Flux/(64, 64, 3, 128)",
            "value": 10474667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12643680\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/SimpleChains/(64, 64, 3, 128)",
            "value": 6391906,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 16)",
            "value": 687400709,
            "unit": "ns",
            "extra": "gctime=0\nmemory=317744848\nallocs=3522\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 64)",
            "value": 2584797531,
            "unit": "ns",
            "extra": "gctime=57045968\nmemory=764442128\nallocs=3522\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 2)",
            "value": 146686907,
            "unit": "ns",
            "extra": "gctime=0\nmemory=187459904\nallocs=3494\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 16)",
            "value": 835907972,
            "unit": "ns",
            "extra": "gctime=61913604\nmemory=659570112\nallocs=10889\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 64)",
            "value": 2939793372,
            "unit": "ns",
            "extra": "gctime=330989507\nmemory=1324513984\nallocs=10889\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 2)",
            "value": 233915123.5,
            "unit": "ns",
            "extra": "gctime=28865943.5\nmemory=465628368\nallocs=10848\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 16)",
            "value": 719291485,
            "unit": "ns",
            "extra": "gctime=3089284\nmemory=390042496\nallocs=10658\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 64)",
            "value": 2463276943,
            "unit": "ns",
            "extra": "gctime=57472935\nmemory=1052224096\nallocs=10658\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 2)",
            "value": 128936852,
            "unit": "ns",
            "extra": "gctime=0\nmemory=196908704\nallocs=10618\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 16)",
            "value": 173926218.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=51375328\nallocs=1042\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 64)",
            "value": 641427082.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=165115008\nallocs=1042\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 2)",
            "value": 45126040,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18201744\nallocs=1034\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 16)",
            "value": 164566019.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=51381088\nallocs=1087\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 64)",
            "value": 640672701.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=165120768\nallocs=1087\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 2)",
            "value": 30365218,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18207504\nallocs=1079\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 16)",
            "value": 187223203,
            "unit": "ns",
            "extra": "gctime=0\nmemory=69608176\nallocs=968\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 64)",
            "value": 717166928.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=238006832\nallocs=968\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 2)",
            "value": 37531607,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20492560\nallocs=957\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 64, 128)",
            "value": 1284443577.5,
            "unit": "ns",
            "extra": "gctime=185882114.5\nmemory=278646368\nallocs=88\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Zygote/(64, 64, 64, 128)",
            "value": 1886176230,
            "unit": "ns",
            "extra": "gctime=3634100\nmemory=430580128\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Tracker/(64, 64, 64, 128)",
            "value": 2370442825,
            "unit": "ns",
            "extra": "gctime=385925596\nmemory=1455080072\nallocs=302\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/ReverseDiff/(64, 64, 64, 128)",
            "value": 2506018476,
            "unit": "ns",
            "extra": "gctime=348485188\nmemory=942830496\nallocs=197\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Flux/(64, 64, 64, 128)",
            "value": 1846056793.5,
            "unit": "ns",
            "extra": "gctime=1642427\nmemory=556546960\nallocs=156\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Enzyme/(64, 64, 64, 128)",
            "value": 2139447150,
            "unit": "ns",
            "extra": "gctime=0\nmemory=430580640\nallocs=140\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/NamedTuple/(64, 64, 64, 128)",
            "value": 340154577,
            "unit": "ns",
            "extra": "gctime=0\nmemory=143677888\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/ComponentArray/(64, 64, 64, 128)",
            "value": 335652065,
            "unit": "ns",
            "extra": "gctime=0\nmemory=143677904\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/Flux/(64, 64, 64, 128)",
            "value": 409343224,
            "unit": "ns",
            "extra": "gctime=0\nmemory=269638112\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 1, 128)",
            "value": 12414313,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4360336\nallocs=87\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Zygote/(64, 64, 1, 128)",
            "value": 18263680,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6736912\nallocs=131\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Tracker/(64, 64, 1, 128)",
            "value": 19176169.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22747992\nallocs=299\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/ReverseDiff/(64, 64, 1, 128)",
            "value": 23973062,
            "unit": "ns",
            "extra": "gctime=0\nmemory=14743104\nallocs=195\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Flux/(64, 64, 1, 128)",
            "value": 18043287.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8711680\nallocs=155\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/SimpleChains/(64, 64, 1, 128)",
            "value": 1172576,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1970896\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Enzyme/(64, 64, 1, 128)",
            "value": 23128665,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6737680\nallocs=139\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/NamedTuple/(64, 64, 1, 128)",
            "value": 2293606,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2249472\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/ComponentArray/(64, 64, 1, 128)",
            "value": 2223750.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2249488\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/Flux/(64, 64, 1, 128)",
            "value": 2071336.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4217632\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/SimpleChains/(64, 64, 1, 128)",
            "value": 199072,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/ReverseDiff (compiled)/(200, 128)",
            "value": 294138.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102672\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Zygote/(200, 128)",
            "value": 267218,
            "unit": "ns",
            "extra": "gctime=0\nmemory=470896\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Tracker/(200, 128)",
            "value": 373977,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1614552\nallocs=149\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/ReverseDiff/(200, 128)",
            "value": 413580,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1040224\nallocs=66\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Flux/(200, 128)",
            "value": 277056.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=571712\nallocs=33\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/SimpleChains/(200, 128)",
            "value": 411307,
            "unit": "ns",
            "extra": "gctime=0\nmemory=747872\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Enzyme/(200, 128)",
            "value": 393665,
            "unit": "ns",
            "extra": "gctime=0\nmemory=205040\nallocs=7\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/NamedTuple/(200, 128)",
            "value": 82103,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102528\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/ComponentArray/(200, 128)",
            "value": 85350,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102528\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/Flux/(200, 128)",
            "value": 87133,
            "unit": "ns",
            "extra": "gctime=0\nmemory=204896\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/SimpleChains/(200, 128)",
            "value": 104495,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 16, 128)",
            "value": 202252009,
            "unit": "ns",
            "extra": "gctime=0\nmemory=69640608\nallocs=87\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Zygote/(64, 64, 16, 128)",
            "value": 332392033,
            "unit": "ns",
            "extra": "gctime=0\nmemory=107626016\nallocs=131\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Tracker/(64, 64, 16, 128)",
            "value": 424344708,
            "unit": "ns",
            "extra": "gctime=1575496.5\nmemory=363701768\nallocs=299\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/ReverseDiff/(64, 64, 16, 128)",
            "value": 489496373,
            "unit": "ns",
            "extra": "gctime=0\nmemory=235664480\nallocs=195\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Flux/(64, 64, 16, 128)",
            "value": 356550409.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=139122704\nallocs=155\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/SimpleChains/(64, 64, 16, 128)",
            "value": 333146263,
            "unit": "ns",
            "extra": "gctime=0\nmemory=31520784\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Enzyme/(64, 64, 16, 128)",
            "value": 473522850,
            "unit": "ns",
            "extra": "gctime=0\nmemory=107626720\nallocs=139\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/NamedTuple/(64, 64, 16, 128)",
            "value": 47574633,
            "unit": "ns",
            "extra": "gctime=0\nmemory=35922880\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/ComponentArray/(64, 64, 16, 128)",
            "value": 47053944,
            "unit": "ns",
            "extra": "gctime=0\nmemory=35922896\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/Flux/(64, 64, 16, 128)",
            "value": 50070117.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=67412960\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/SimpleChains/(64, 64, 16, 128)",
            "value": 28487818,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/ReverseDiff (compiled)/(2000, 128)",
            "value": 19266122.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024272\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Zygote/(2000, 128)",
            "value": 19780999,
            "unit": "ns",
            "extra": "gctime=0\nmemory=19082928\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Tracker/(2000, 128)",
            "value": 23815988.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=59293848\nallocs=149\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/ReverseDiff/(2000, 128)",
            "value": 24328162,
            "unit": "ns",
            "extra": "gctime=0\nmemory=39178656\nallocs=66\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Flux/(2000, 128)",
            "value": 19755229,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20105344\nallocs=33\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Enzyme/(2000, 128)",
            "value": 21032811.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2048240\nallocs=7\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/NamedTuple/(2000, 128)",
            "value": 6550793,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024128\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/ComponentArray/(2000, 128)",
            "value": 6543098,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024128\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/Flux/(2000, 128)",
            "value": 6530683.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2048096\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "avik.pal.2017@gmail.com",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "distinct": true,
          "id": "b9220f67cabcd5a43a3112473de9b9d5783b54d3",
          "message": "test: Bidirectional Enzyme fails only on Mac",
          "timestamp": "2024-07-29T07:19:15-07:00",
          "tree_id": "7aefd19b3d1f6d81c1939777e4e6cab283f38d1a",
          "url": "https://github.com/LuxDL/Lux.jl/commit/b9220f67cabcd5a43a3112473de9b9d5783b54d3"
        },
        "date": 1722264888190,
        "tool": "julia",
        "benches": [
          {
            "name": "Dense(2 => 2)/cpu/reverse/ReverseDiff (compiled)/(2, 128)",
            "value": 3581.1111111111113,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":9,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Zygote/(2, 128)",
            "value": 7292.142857142857,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6016\nallocs=42\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Tracker/(2, 128)",
            "value": 20097,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17224\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/ReverseDiff/(2, 128)",
            "value": 9631.333333333334,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9968\nallocs=59\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Flux/(2, 128)",
            "value": 8833.166666666668,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5472\nallocs=28\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":3,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/SimpleChains/(2, 128)",
            "value": 4240.222222222223,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3120\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":9,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Enzyme/(2, 128)",
            "value": 4509.625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2320\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/NamedTuple/(2, 128)",
            "value": 1110.0727272727272,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1168\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":165,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/ComponentArray/(2, 128)",
            "value": 1190.345864661654,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1168\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":133,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/Flux/(2, 128)",
            "value": 1772.3186813186812,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2176\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":91,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/SimpleChains/(2, 128)",
            "value": 179.93900709219858,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":705,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/ReverseDiff (compiled)/(20, 128)",
            "value": 17152,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10592\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Zygote/(20, 128)",
            "value": 16911.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=35664\nallocs=42\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Tracker/(20, 128)",
            "value": 36158,
            "unit": "ns",
            "extra": "gctime=0\nmemory=124696\nallocs=135\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/ReverseDiff/(20, 128)",
            "value": 28623,
            "unit": "ns",
            "extra": "gctime=0\nmemory=78432\nallocs=57\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Flux/(20, 128)",
            "value": 19917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44400\nallocs=28\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/SimpleChains/(20, 128)",
            "value": 17112,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17632\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Enzyme/(20, 128)",
            "value": 25227,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20880\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/NamedTuple/(20, 128)",
            "value": 3810.875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10448\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/ComponentArray/(20, 128)",
            "value": 3894.75,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10448\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/Flux/(20, 128)",
            "value": 4668.714285714285,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20736\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/SimpleChains/(20, 128)",
            "value": 1660.1,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":10,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 3, 128)",
            "value": 40386360,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13063408\nallocs=87\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Zygote/(64, 64, 3, 128)",
            "value": 58048752.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20187840\nallocs=131\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Tracker/(64, 64, 3, 128)",
            "value": 77355434.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=68205704\nallocs=301\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/ReverseDiff/(64, 64, 3, 128)",
            "value": 92098201,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44197424\nallocs=197\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Flux/(64, 64, 3, 128)",
            "value": 77638697.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=26098864\nallocs=155\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/SimpleChains/(64, 64, 3, 128)",
            "value": 11425372,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5908032\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Enzyme/(64, 64, 3, 128)",
            "value": 85055575,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20188592\nallocs=139\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/NamedTuple/(64, 64, 3, 128)",
            "value": 7652453,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6739264\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/ComponentArray/(64, 64, 3, 128)",
            "value": 7558022,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6739280\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/Flux/(64, 64, 3, 128)",
            "value": 11380260.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12643680\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/SimpleChains/(64, 64, 3, 128)",
            "value": 6370135,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 16)",
            "value": 684722763.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=317744848\nallocs=3522\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 64)",
            "value": 2540907547,
            "unit": "ns",
            "extra": "gctime=60932586\nmemory=764442128\nallocs=3522\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 2)",
            "value": 144208938,
            "unit": "ns",
            "extra": "gctime=0\nmemory=187459904\nallocs=3494\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 16)",
            "value": 892561377,
            "unit": "ns",
            "extra": "gctime=68228513\nmemory=659567936\nallocs=10888\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 64)",
            "value": 2997851620,
            "unit": "ns",
            "extra": "gctime=309526259\nmemory=1324513984\nallocs=10889\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 2)",
            "value": 214273209,
            "unit": "ns",
            "extra": "gctime=3475905\nmemory=465628368\nallocs=10848\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 16)",
            "value": 699542143,
            "unit": "ns",
            "extra": "gctime=2377851\nmemory=390042496\nallocs=10658\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 64)",
            "value": 2629980619,
            "unit": "ns",
            "extra": "gctime=60206098\nmemory=1052224096\nallocs=10658\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 2)",
            "value": 144936446,
            "unit": "ns",
            "extra": "gctime=0\nmemory=196908704\nallocs=10618\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 16)",
            "value": 184361572.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=51375328\nallocs=1042\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 64)",
            "value": 637309089,
            "unit": "ns",
            "extra": "gctime=0\nmemory=165115008\nallocs=1042\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 2)",
            "value": 34506797,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18201744\nallocs=1034\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 16)",
            "value": 162603795,
            "unit": "ns",
            "extra": "gctime=0\nmemory=51381088\nallocs=1087\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 64)",
            "value": 630593315.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=165120768\nallocs=1087\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 2)",
            "value": 29861420,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18207504\nallocs=1079\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 16)",
            "value": 209144364.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=69608176\nallocs=968\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 64)",
            "value": 768864654.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=238006832\nallocs=968\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 2)",
            "value": 35686603.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20492560\nallocs=957\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 64, 128)",
            "value": 1233165049.5,
            "unit": "ns",
            "extra": "gctime=163621470.5\nmemory=278646368\nallocs=88\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Zygote/(64, 64, 64, 128)",
            "value": 1847279770.5,
            "unit": "ns",
            "extra": "gctime=3739145\nmemory=430580128\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Tracker/(64, 64, 64, 128)",
            "value": 2337901195,
            "unit": "ns",
            "extra": "gctime=385662620\nmemory=1455080072\nallocs=302\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/ReverseDiff/(64, 64, 64, 128)",
            "value": 2493177116,
            "unit": "ns",
            "extra": "gctime=314701064\nmemory=942830496\nallocs=197\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Flux/(64, 64, 64, 128)",
            "value": 1895219376.5,
            "unit": "ns",
            "extra": "gctime=1575954.5\nmemory=556546960\nallocs=156\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Enzyme/(64, 64, 64, 128)",
            "value": 2098569612,
            "unit": "ns",
            "extra": "gctime=0\nmemory=430580640\nallocs=140\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/NamedTuple/(64, 64, 64, 128)",
            "value": 327876125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=143677888\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/ComponentArray/(64, 64, 64, 128)",
            "value": 322956041,
            "unit": "ns",
            "extra": "gctime=0\nmemory=143677904\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/Flux/(64, 64, 64, 128)",
            "value": 374715425,
            "unit": "ns",
            "extra": "gctime=0\nmemory=269638112\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 1, 128)",
            "value": 11777394,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4360336\nallocs=87\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Zygote/(64, 64, 1, 128)",
            "value": 17874850,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6736912\nallocs=131\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Tracker/(64, 64, 1, 128)",
            "value": 18840163,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22747992\nallocs=299\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/ReverseDiff/(64, 64, 1, 128)",
            "value": 23632391,
            "unit": "ns",
            "extra": "gctime=0\nmemory=14743104\nallocs=195\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Flux/(64, 64, 1, 128)",
            "value": 17747023,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8711680\nallocs=155\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/SimpleChains/(64, 64, 1, 128)",
            "value": 1147621,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1970896\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Enzyme/(64, 64, 1, 128)",
            "value": 22618101.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6737680\nallocs=139\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/NamedTuple/(64, 64, 1, 128)",
            "value": 2284871,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2249472\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/ComponentArray/(64, 64, 1, 128)",
            "value": 2208200.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2249488\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/Flux/(64, 64, 1, 128)",
            "value": 2053872.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4217632\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/SimpleChains/(64, 64, 1, 128)",
            "value": 196797,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/ReverseDiff (compiled)/(200, 128)",
            "value": 289761,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102672\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Zygote/(200, 128)",
            "value": 263362,
            "unit": "ns",
            "extra": "gctime=0\nmemory=470896\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Tracker/(200, 128)",
            "value": 361575,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1614552\nallocs=149\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/ReverseDiff/(200, 128)",
            "value": 403684,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1040224\nallocs=66\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Flux/(200, 128)",
            "value": 271066,
            "unit": "ns",
            "extra": "gctime=0\nmemory=571712\nallocs=33\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/SimpleChains/(200, 128)",
            "value": 402932.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=747872\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Enzyme/(200, 128)",
            "value": 393309,
            "unit": "ns",
            "extra": "gctime=0\nmemory=205040\nallocs=7\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/NamedTuple/(200, 128)",
            "value": 80280,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102528\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/ComponentArray/(200, 128)",
            "value": 80751,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102528\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/Flux/(200, 128)",
            "value": 85429,
            "unit": "ns",
            "extra": "gctime=0\nmemory=204896\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/SimpleChains/(200, 128)",
            "value": 105027,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 16, 128)",
            "value": 198991930,
            "unit": "ns",
            "extra": "gctime=0\nmemory=69640608\nallocs=87\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Zygote/(64, 64, 16, 128)",
            "value": 326171871,
            "unit": "ns",
            "extra": "gctime=0\nmemory=107626016\nallocs=131\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Tracker/(64, 64, 16, 128)",
            "value": 414427316.5,
            "unit": "ns",
            "extra": "gctime=1989513\nmemory=363701768\nallocs=299\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/ReverseDiff/(64, 64, 16, 128)",
            "value": 433879078.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=235664480\nallocs=195\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Flux/(64, 64, 16, 128)",
            "value": 409918522,
            "unit": "ns",
            "extra": "gctime=0\nmemory=139122704\nallocs=155\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/SimpleChains/(64, 64, 16, 128)",
            "value": 316295915,
            "unit": "ns",
            "extra": "gctime=0\nmemory=31520784\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Enzyme/(64, 64, 16, 128)",
            "value": 494850601.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=107626720\nallocs=139\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/NamedTuple/(64, 64, 16, 128)",
            "value": 47336289,
            "unit": "ns",
            "extra": "gctime=0\nmemory=35922880\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/ComponentArray/(64, 64, 16, 128)",
            "value": 46607692,
            "unit": "ns",
            "extra": "gctime=0\nmemory=35922896\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/Flux/(64, 64, 16, 128)",
            "value": 57048314.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=67412960\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/SimpleChains/(64, 64, 16, 128)",
            "value": 27300249,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/ReverseDiff (compiled)/(2000, 128)",
            "value": 18698468.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024272\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Zygote/(2000, 128)",
            "value": 19451830,
            "unit": "ns",
            "extra": "gctime=0\nmemory=19082928\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Tracker/(2000, 128)",
            "value": 22994195,
            "unit": "ns",
            "extra": "gctime=0\nmemory=59293848\nallocs=149\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/ReverseDiff/(2000, 128)",
            "value": 24000007.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=39178656\nallocs=66\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Flux/(2000, 128)",
            "value": 19548054,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20105344\nallocs=33\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Enzyme/(2000, 128)",
            "value": 20691287,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2048240\nallocs=7\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/NamedTuple/(2000, 128)",
            "value": 6468799,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024128\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/ComponentArray/(2000, 128)",
            "value": 6436489.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024128\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/Flux/(2000, 128)",
            "value": 6460534,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2048096\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "59402fe726a88e20bd80f31cb408c02c8581b7af",
          "message": "chore: revert version bump",
          "timestamp": "2024-07-29T07:28:55-07:00",
          "tree_id": "9b362dc50fc1ab1bb7222cfbabe104358d85b3f9",
          "url": "https://github.com/LuxDL/Lux.jl/commit/59402fe726a88e20bd80f31cb408c02c8581b7af"
        },
        "date": 1722267082516,
        "tool": "julia",
        "benches": [
          {
            "name": "Dense(2 => 2)/cpu/reverse/ReverseDiff (compiled)/(2, 128)",
            "value": 3654.375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Zygote/(2, 128)",
            "value": 7423.833333333333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6016\nallocs=42\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Tracker/(2, 128)",
            "value": 20889,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17224\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/ReverseDiff/(2, 128)",
            "value": 9879,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9968\nallocs=59\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Flux/(2, 128)",
            "value": 9033,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5472\nallocs=28\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":5,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/SimpleChains/(2, 128)",
            "value": 4503.375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3120\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Enzyme/(2, 128)",
            "value": 4685,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2320\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/NamedTuple/(2, 128)",
            "value": 1129.4184397163122,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1168\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":141,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/ComponentArray/(2, 128)",
            "value": 1183.8444444444444,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1168\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":135,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/Flux/(2, 128)",
            "value": 1806.017543859649,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2176\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":57,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/SimpleChains/(2, 128)",
            "value": 180.11344537815125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":714,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/ReverseDiff (compiled)/(20, 128)",
            "value": 17302,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10592\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Zygote/(20, 128)",
            "value": 17042,
            "unit": "ns",
            "extra": "gctime=0\nmemory=35664\nallocs=42\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Tracker/(20, 128)",
            "value": 36989,
            "unit": "ns",
            "extra": "gctime=0\nmemory=124696\nallocs=135\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/ReverseDiff/(20, 128)",
            "value": 29144,
            "unit": "ns",
            "extra": "gctime=0\nmemory=78432\nallocs=57\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Flux/(20, 128)",
            "value": 21400,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44400\nallocs=28\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/SimpleChains/(20, 128)",
            "value": 17453,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17632\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Enzyme/(20, 128)",
            "value": 25457,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20880\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/NamedTuple/(20, 128)",
            "value": 3867.8125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10448\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/ComponentArray/(20, 128)",
            "value": 3941.125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10448\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/Flux/(20, 128)",
            "value": 4884.928571428572,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20736\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/SimpleChains/(20, 128)",
            "value": 1651.1,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":10,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 3, 128)",
            "value": 38881796,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13063408\nallocs=87\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Zygote/(64, 64, 3, 128)",
            "value": 58827311,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20187840\nallocs=131\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Tracker/(64, 64, 3, 128)",
            "value": 67404189,
            "unit": "ns",
            "extra": "gctime=0\nmemory=68205704\nallocs=301\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/ReverseDiff/(64, 64, 3, 128)",
            "value": 90726308,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44197424\nallocs=197\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Flux/(64, 64, 3, 128)",
            "value": 72685526,
            "unit": "ns",
            "extra": "gctime=0\nmemory=26098864\nallocs=155\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/SimpleChains/(64, 64, 3, 128)",
            "value": 12054419,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5908032\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Enzyme/(64, 64, 3, 128)",
            "value": 88616409,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20188592\nallocs=139\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/NamedTuple/(64, 64, 3, 128)",
            "value": 7678423,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6739264\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/ComponentArray/(64, 64, 3, 128)",
            "value": 7587021,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6739280\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/Flux/(64, 64, 3, 128)",
            "value": 10033851,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12643680\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/SimpleChains/(64, 64, 3, 128)",
            "value": 6401934.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 16)",
            "value": 695594050,
            "unit": "ns",
            "extra": "gctime=0\nmemory=317744848\nallocs=3522\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 64)",
            "value": 2591199265,
            "unit": "ns",
            "extra": "gctime=59956669\nmemory=764442128\nallocs=3522\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 2)",
            "value": 145450054,
            "unit": "ns",
            "extra": "gctime=0\nmemory=187459904\nallocs=3494\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 16)",
            "value": 856061822,
            "unit": "ns",
            "extra": "gctime=63689580\nmemory=659567936\nallocs=10888\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 64)",
            "value": 2884249763,
            "unit": "ns",
            "extra": "gctime=338494774\nmemory=1324513984\nallocs=10889\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 2)",
            "value": 246325046.5,
            "unit": "ns",
            "extra": "gctime=30160524\nmemory=465628368\nallocs=10848\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 16)",
            "value": 714210079,
            "unit": "ns",
            "extra": "gctime=3116620.5\nmemory=390042496\nallocs=10658\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 64)",
            "value": 2711805409,
            "unit": "ns",
            "extra": "gctime=227655394\nmemory=1052224096\nallocs=10658\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 2)",
            "value": 127679277,
            "unit": "ns",
            "extra": "gctime=0\nmemory=196908704\nallocs=10618\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 16)",
            "value": 172995580.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=51375328\nallocs=1042\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 64)",
            "value": 651366871,
            "unit": "ns",
            "extra": "gctime=0\nmemory=165115008\nallocs=1042\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 2)",
            "value": 45176285.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18201744\nallocs=1034\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 16)",
            "value": 164593550,
            "unit": "ns",
            "extra": "gctime=0\nmemory=51381088\nallocs=1087\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 64)",
            "value": 640463071,
            "unit": "ns",
            "extra": "gctime=0\nmemory=165120768\nallocs=1087\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 2)",
            "value": 30093452,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18207504\nallocs=1079\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 16)",
            "value": 204835624,
            "unit": "ns",
            "extra": "gctime=0\nmemory=69608176\nallocs=968\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 64)",
            "value": 720482648.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=238006832\nallocs=968\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 2)",
            "value": 35672885,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20492560\nallocs=957\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 64, 128)",
            "value": 1309190313,
            "unit": "ns",
            "extra": "gctime=173668299\nmemory=278646368\nallocs=88\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Zygote/(64, 64, 64, 128)",
            "value": 1883793155,
            "unit": "ns",
            "extra": "gctime=3985667\nmemory=430580128\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Tracker/(64, 64, 64, 128)",
            "value": 2353304817,
            "unit": "ns",
            "extra": "gctime=404827310\nmemory=1455080072\nallocs=302\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/ReverseDiff/(64, 64, 64, 128)",
            "value": 2438276652,
            "unit": "ns",
            "extra": "gctime=341519028\nmemory=942830496\nallocs=197\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Flux/(64, 64, 64, 128)",
            "value": 1867266298,
            "unit": "ns",
            "extra": "gctime=2023380.5\nmemory=556546960\nallocs=156\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Enzyme/(64, 64, 64, 128)",
            "value": 2034616539,
            "unit": "ns",
            "extra": "gctime=0\nmemory=430580640\nallocs=140\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/NamedTuple/(64, 64, 64, 128)",
            "value": 335394624.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=143677888\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/ComponentArray/(64, 64, 64, 128)",
            "value": 332856663,
            "unit": "ns",
            "extra": "gctime=0\nmemory=143677904\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/Flux/(64, 64, 64, 128)",
            "value": 459008363,
            "unit": "ns",
            "extra": "gctime=0\nmemory=269638112\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 1, 128)",
            "value": 11905264.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4360336\nallocs=87\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Zygote/(64, 64, 1, 128)",
            "value": 18205658.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6736912\nallocs=131\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Tracker/(64, 64, 1, 128)",
            "value": 19288136,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22747992\nallocs=299\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/ReverseDiff/(64, 64, 1, 128)",
            "value": 23962152.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=14743104\nallocs=195\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Flux/(64, 64, 1, 128)",
            "value": 18014189,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8711680\nallocs=155\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/SimpleChains/(64, 64, 1, 128)",
            "value": 1167993,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1970896\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Enzyme/(64, 64, 1, 128)",
            "value": 23074801.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6737680\nallocs=139\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/NamedTuple/(64, 64, 1, 128)",
            "value": 2441973,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2249472\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/ComponentArray/(64, 64, 1, 128)",
            "value": 2232336,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2249488\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/Flux/(64, 64, 1, 128)",
            "value": 2074976,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4217632\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/SimpleChains/(64, 64, 1, 128)",
            "value": 199804,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/ReverseDiff (compiled)/(200, 128)",
            "value": 293199.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102672\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Zygote/(200, 128)",
            "value": 266209,
            "unit": "ns",
            "extra": "gctime=0\nmemory=470896\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Tracker/(200, 128)",
            "value": 370003,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1614552\nallocs=149\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/ReverseDiff/(200, 128)",
            "value": 411882,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1040224\nallocs=66\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Flux/(200, 128)",
            "value": 275706.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=571712\nallocs=33\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/SimpleChains/(200, 128)",
            "value": 409858,
            "unit": "ns",
            "extra": "gctime=0\nmemory=747872\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Enzyme/(200, 128)",
            "value": 396372,
            "unit": "ns",
            "extra": "gctime=0\nmemory=205040\nallocs=7\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/NamedTuple/(200, 128)",
            "value": 81443,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102528\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/ComponentArray/(200, 128)",
            "value": 82123,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102528\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/Flux/(200, 128)",
            "value": 87013,
            "unit": "ns",
            "extra": "gctime=0\nmemory=204896\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/SimpleChains/(200, 128)",
            "value": 104516,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 16, 128)",
            "value": 197115275,
            "unit": "ns",
            "extra": "gctime=0\nmemory=69640608\nallocs=87\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Zygote/(64, 64, 16, 128)",
            "value": 331248547.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=107626016\nallocs=131\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Tracker/(64, 64, 16, 128)",
            "value": 427260324,
            "unit": "ns",
            "extra": "gctime=1845525.5\nmemory=363701768\nallocs=299\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/ReverseDiff/(64, 64, 16, 128)",
            "value": 483420905,
            "unit": "ns",
            "extra": "gctime=0\nmemory=235664480\nallocs=195\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Flux/(64, 64, 16, 128)",
            "value": 386517355,
            "unit": "ns",
            "extra": "gctime=0\nmemory=139122704\nallocs=155\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/SimpleChains/(64, 64, 16, 128)",
            "value": 338431242,
            "unit": "ns",
            "extra": "gctime=0\nmemory=31520784\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Enzyme/(64, 64, 16, 128)",
            "value": 469112435,
            "unit": "ns",
            "extra": "gctime=0\nmemory=107626720\nallocs=139\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/NamedTuple/(64, 64, 16, 128)",
            "value": 47409570,
            "unit": "ns",
            "extra": "gctime=0\nmemory=35922880\nallocs=49\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/ComponentArray/(64, 64, 16, 128)",
            "value": 46895813,
            "unit": "ns",
            "extra": "gctime=0\nmemory=35922896\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/Flux/(64, 64, 16, 128)",
            "value": 56448294.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=67412960\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/SimpleChains/(64, 64, 16, 128)",
            "value": 28438815,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/ReverseDiff (compiled)/(2000, 128)",
            "value": 19130483.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024272\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Zygote/(2000, 128)",
            "value": 19606525,
            "unit": "ns",
            "extra": "gctime=0\nmemory=19082928\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Tracker/(2000, 128)",
            "value": 23552054,
            "unit": "ns",
            "extra": "gctime=0\nmemory=59293848\nallocs=149\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/ReverseDiff/(2000, 128)",
            "value": 24232581.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=39178656\nallocs=66\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Flux/(2000, 128)",
            "value": 19724625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20105344\nallocs=33\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Enzyme/(2000, 128)",
            "value": 20980827,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2048240\nallocs=7\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/NamedTuple/(2000, 128)",
            "value": 6536465,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024128\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/ComponentArray/(2000, 128)",
            "value": 6533604,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024128\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/Flux/(2000, 128)",
            "value": 6545116,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2048096\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "avik.pal.2017@gmail.com",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "distinct": true,
          "id": "9e7c74e6dfc83e372e79c8adbe704111ecdfaea2",
          "message": "docs: add new function to docs",
          "timestamp": "2024-08-01T17:13:38-07:00",
          "tree_id": "9fefaf1ef6a41d19f28409ac07da20ead5e7ff1e",
          "url": "https://github.com/LuxDL/Lux.jl/commit/9e7c74e6dfc83e372e79c8adbe704111ecdfaea2"
        },
        "date": 1722562368524,
        "tool": "julia",
        "benches": [
          {
            "name": "Dense(2 => 2)/cpu/reverse/ReverseDiff (compiled)/(2, 128)",
            "value": 3664.375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Zygote/(2, 128)",
            "value": 6756.857142857143,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6016\nallocs=42\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Tracker/(2, 128)",
            "value": 21019,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17224\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/ReverseDiff/(2, 128)",
            "value": 10110.8,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9968\nallocs=59\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":5,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Flux/(2, 128)",
            "value": 9363.6,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5472\nallocs=28\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":5,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/SimpleChains/(2, 128)",
            "value": 4504.625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3120\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Enzyme/(2, 128)",
            "value": 4961.75,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2688\nallocs=6\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/NamedTuple/(2, 128)",
            "value": 1043.1168831168832,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1168\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":154,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/ComponentArray/(2, 128)",
            "value": 1060.5555555555557,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1168\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":162,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/Flux/(2, 128)",
            "value": 1866.423076923077,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2176\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":39,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/SimpleChains/(2, 128)",
            "value": 180.21161825726142,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":723,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/ReverseDiff (compiled)/(20, 128)",
            "value": 17762.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10592\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Zygote/(20, 128)",
            "value": 13435,
            "unit": "ns",
            "extra": "gctime=0\nmemory=35664\nallocs=42\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Tracker/(20, 128)",
            "value": 39324,
            "unit": "ns",
            "extra": "gctime=0\nmemory=124696\nallocs=135\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/ReverseDiff/(20, 128)",
            "value": 30397,
            "unit": "ns",
            "extra": "gctime=0\nmemory=78432\nallocs=57\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Flux/(20, 128)",
            "value": 20418,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44400\nallocs=28\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/SimpleChains/(20, 128)",
            "value": 17843,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17632\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Enzyme/(20, 128)",
            "value": 25748,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21248\nallocs=6\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/NamedTuple/(20, 128)",
            "value": 1465.7,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10448\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":10,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/ComponentArray/(20, 128)",
            "value": 1506.8,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10448\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":10,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/Flux/(20, 128)",
            "value": 4840.428571428572,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20736\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/SimpleChains/(20, 128)",
            "value": 1661.1,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":10,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 3, 128)",
            "value": 90294143,
            "unit": "ns",
            "extra": "gctime=0\nmemory=14310896\nallocs=97\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Zygote/(64, 64, 3, 128)",
            "value": 77203381,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22682688\nallocs=153\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Tracker/(64, 64, 3, 128)",
            "value": 160594802,
            "unit": "ns",
            "extra": "gctime=0\nmemory=70700488\nallocs=321\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/ReverseDiff/(64, 64, 3, 128)",
            "value": 190660746,
            "unit": "ns",
            "extra": "gctime=0\nmemory=46692208\nallocs=217\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Flux/(64, 64, 3, 128)",
            "value": 153189129,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28593648\nallocs=175\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/SimpleChains/(64, 64, 3, 128)",
            "value": 11984206.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5908032\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Enzyme/(64, 64, 3, 128)",
            "value": 230386017.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22683504\nallocs=173\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/NamedTuple/(64, 64, 3, 128)",
            "value": 15634506,
            "unit": "ns",
            "extra": "gctime=0\nmemory=7986720\nallocs=61\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/ComponentArray/(64, 64, 3, 128)",
            "value": 15538928,
            "unit": "ns",
            "extra": "gctime=0\nmemory=7986960\nallocs=60\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/Flux/(64, 64, 3, 128)",
            "value": 31162360,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13891072\nallocs=60\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/SimpleChains/(64, 64, 3, 128)",
            "value": 6399474,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 16)",
            "value": 1068772497.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=357822480\nallocs=3700\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 64)",
            "value": 2995840802,
            "unit": "ns",
            "extra": "gctime=59943511\nmemory=804519760\nallocs=3700\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 2)",
            "value": 164886542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=227486784\nallocs=3412\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 16)",
            "value": 1384507856,
            "unit": "ns",
            "extra": "gctime=2705666\nmemory=699655168\nallocs=11149\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 64)",
            "value": 3671340191,
            "unit": "ns",
            "extra": "gctime=86436414\nmemory=1364596864\nallocs=11148\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 2)",
            "value": 389888559,
            "unit": "ns",
            "extra": "gctime=2793860\nmemory=505662672\nallocs=10848\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 16)",
            "value": 1518928884,
            "unit": "ns",
            "extra": "gctime=0\nmemory=430127552\nallocs=10918\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 64)",
            "value": 4247770691,
            "unit": "ns",
            "extra": "gctime=81840534\nmemory=1092309152\nallocs=10918\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 2)",
            "value": 401207499.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=236943008\nallocs=10618\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 16)",
            "value": 371371334,
            "unit": "ns",
            "extra": "gctime=0\nmemory=71415584\nallocs=1142\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 64)",
            "value": 926409352,
            "unit": "ns",
            "extra": "gctime=0\nmemory=185155264\nallocs=1142\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 2)",
            "value": 51293524,
            "unit": "ns",
            "extra": "gctime=0\nmemory=38216624\nallocs=1004\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 16)",
            "value": 361983037,
            "unit": "ns",
            "extra": "gctime=0\nmemory=71426944\nallocs=1269\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 64)",
            "value": 928505021.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=185166624\nallocs=1269\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 2)",
            "value": 51045376,
            "unit": "ns",
            "extra": "gctime=0\nmemory=38226736\nallocs=1131\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 16)",
            "value": 530578546,
            "unit": "ns",
            "extra": "gctime=0\nmemory=89650704\nallocs=1098\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 64)",
            "value": 1496859062,
            "unit": "ns",
            "extra": "gctime=0\nmemory=258049360\nallocs=1098\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 2)",
            "value": 180025378.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=40509712\nallocs=957\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 64, 128)",
            "value": 1172729784,
            "unit": "ns",
            "extra": "gctime=0\nmemory=305218144\nallocs=98\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Zygote/(64, 64, 64, 128)",
            "value": 1581046876,
            "unit": "ns",
            "extra": "gctime=1075710.5\nmemory=483723552\nallocs=154\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Tracker/(64, 64, 64, 128)",
            "value": 2672787327,
            "unit": "ns",
            "extra": "gctime=433284900\nmemory=1508223432\nallocs=322\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/ReverseDiff/(64, 64, 64, 128)",
            "value": 2491782332,
            "unit": "ns",
            "extra": "gctime=181439792\nmemory=995973856\nallocs=217\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Flux/(64, 64, 64, 128)",
            "value": 2160602259.5,
            "unit": "ns",
            "extra": "gctime=3583371\nmemory=609690320\nallocs=176\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Enzyme/(64, 64, 64, 128)",
            "value": 2211993600,
            "unit": "ns",
            "extra": "gctime=0\nmemory=483724144\nallocs=175\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/NamedTuple/(64, 64, 64, 128)",
            "value": 285674800,
            "unit": "ns",
            "extra": "gctime=0\nmemory=170249632\nallocs=61\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/ComponentArray/(64, 64, 64, 128)",
            "value": 281190731,
            "unit": "ns",
            "extra": "gctime=0\nmemory=170249872\nallocs=60\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/Flux/(64, 64, 64, 128)",
            "value": 457207860,
            "unit": "ns",
            "extra": "gctime=0\nmemory=296209792\nallocs=60\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 1, 128)",
            "value": 11786564,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4777552\nallocs=97\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Zygote/(64, 64, 1, 128)",
            "value": 35061902.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=7571216\nallocs=153\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Tracker/(64, 64, 1, 128)",
            "value": 16699499,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23582232\nallocs=319\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/ReverseDiff/(64, 64, 1, 128)",
            "value": 21176534,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15577344\nallocs=215\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Flux/(64, 64, 1, 128)",
            "value": 15482828,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9545920\nallocs=175\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/SimpleChains/(64, 64, 1, 128)",
            "value": 1162177,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1970896\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Enzyme/(64, 64, 1, 128)",
            "value": 36352967,
            "unit": "ns",
            "extra": "gctime=0\nmemory=7572016\nallocs=171\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/NamedTuple/(64, 64, 1, 128)",
            "value": 4544292,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2666656\nallocs=61\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/ComponentArray/(64, 64, 1, 128)",
            "value": 4525478,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2666896\nallocs=60\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/Flux/(64, 64, 1, 128)",
            "value": 2044481.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4634752\nallocs=60\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/SimpleChains/(64, 64, 1, 128)",
            "value": 203549,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/ReverseDiff (compiled)/(200, 128)",
            "value": 384336.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102672\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Zygote/(200, 128)",
            "value": 210483,
            "unit": "ns",
            "extra": "gctime=0\nmemory=470896\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Tracker/(200, 128)",
            "value": 389086,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1614552\nallocs=149\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/ReverseDiff/(200, 128)",
            "value": 522203,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1040224\nallocs=66\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Flux/(200, 128)",
            "value": 298136,
            "unit": "ns",
            "extra": "gctime=0\nmemory=571712\nallocs=33\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/SimpleChains/(200, 128)",
            "value": 415635,
            "unit": "ns",
            "extra": "gctime=0\nmemory=747872\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Enzyme/(200, 128)",
            "value": 433097,
            "unit": "ns",
            "extra": "gctime=0\nmemory=205408\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/NamedTuple/(200, 128)",
            "value": 60222,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102528\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/ComponentArray/(200, 128)",
            "value": 61455,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102528\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/Flux/(200, 128)",
            "value": 93985,
            "unit": "ns",
            "extra": "gctime=0\nmemory=204896\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/SimpleChains/(200, 128)",
            "value": 105287,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 16, 128)",
            "value": 323291615,
            "unit": "ns",
            "extra": "gctime=0\nmemory=76285088\nallocs=97\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Zygote/(64, 64, 16, 128)",
            "value": 287830703,
            "unit": "ns",
            "extra": "gctime=0\nmemory=120914848\nallocs=153\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Tracker/(64, 64, 16, 128)",
            "value": 574565922.5,
            "unit": "ns",
            "extra": "gctime=1941565.5\nmemory=376990536\nallocs=319\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/ReverseDiff/(64, 64, 16, 128)",
            "value": 626753211,
            "unit": "ns",
            "extra": "gctime=0\nmemory=248953248\nallocs=215\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Flux/(64, 64, 16, 128)",
            "value": 566033762.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=152411472\nallocs=175\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/SimpleChains/(64, 64, 16, 128)",
            "value": 330579312.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=31520784\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Enzyme/(64, 64, 16, 128)",
            "value": 665801500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=120915632\nallocs=174\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/NamedTuple/(64, 64, 16, 128)",
            "value": 39671362,
            "unit": "ns",
            "extra": "gctime=0\nmemory=42567328\nallocs=61\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/ComponentArray/(64, 64, 16, 128)",
            "value": 39786697,
            "unit": "ns",
            "extra": "gctime=0\nmemory=42567568\nallocs=60\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/Flux/(64, 64, 16, 128)",
            "value": 99957123.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=74057344\nallocs=60\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/SimpleChains/(64, 64, 16, 128)",
            "value": 28529013.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/ReverseDiff (compiled)/(2000, 128)",
            "value": 21204339,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024272\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Zygote/(2000, 128)",
            "value": 21519238.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=19082928\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Tracker/(2000, 128)",
            "value": 22990955.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=59293848\nallocs=149\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/ReverseDiff/(2000, 128)",
            "value": 26414542.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=39178656\nallocs=66\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Flux/(2000, 128)",
            "value": 19330922.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20105344\nallocs=33\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Enzyme/(2000, 128)",
            "value": 20892108,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2048608\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/NamedTuple/(2000, 128)",
            "value": 6247196,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024128\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/ComponentArray/(2000, 128)",
            "value": 6014933,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024128\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/Flux/(2000, 128)",
            "value": 6473393,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2048096\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "avik.pal.2017@gmail.com",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "distinct": true,
          "id": "22b87b57e77741542d3f593f931bb1ab5a86273b",
          "message": "fix: update `DynamicExpressionsLayer` to support new versions",
          "timestamp": "2024-08-01T18:56:20-07:00",
          "tree_id": "4e6b043c8d60ff1f87107e1666b744cb60118c77",
          "url": "https://github.com/LuxDL/Lux.jl/commit/22b87b57e77741542d3f593f931bb1ab5a86273b"
        },
        "date": 1722567094141,
        "tool": "julia",
        "benches": [
          {
            "name": "Dense(2 => 2)/cpu/reverse/ReverseDiff (compiled)/(2, 128)",
            "value": 3678.125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Zygote/(2, 128)",
            "value": 6732.571428571428,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6016\nallocs=42\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Tracker/(2, 128)",
            "value": 21270,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17224\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/ReverseDiff/(2, 128)",
            "value": 9706,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9968\nallocs=59\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":5,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Flux/(2, 128)",
            "value": 9059,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5472\nallocs=28\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":5,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/SimpleChains/(2, 128)",
            "value": 4460.875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3120\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Enzyme/(2, 128)",
            "value": 4948.6875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2688\nallocs=6\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/NamedTuple/(2, 128)",
            "value": 1034.9936305732483,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1168\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":157,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/ComponentArray/(2, 128)",
            "value": 1058.9415204678362,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1168\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":171,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/Flux/(2, 128)",
            "value": 1797.0892857142858,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2176\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":56,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/SimpleChains/(2, 128)",
            "value": 180.6596638655462,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":714,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/ReverseDiff (compiled)/(20, 128)",
            "value": 17252,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10592\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Zygote/(20, 128)",
            "value": 12543,
            "unit": "ns",
            "extra": "gctime=0\nmemory=35664\nallocs=42\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Tracker/(20, 128)",
            "value": 39294,
            "unit": "ns",
            "extra": "gctime=0\nmemory=124696\nallocs=135\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/ReverseDiff/(20, 128)",
            "value": 28989.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=78432\nallocs=57\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Flux/(20, 128)",
            "value": 20068,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44400\nallocs=28\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/SimpleChains/(20, 128)",
            "value": 17102,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17632\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Enzyme/(20, 128)",
            "value": 25538,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21248\nallocs=6\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/NamedTuple/(20, 128)",
            "value": 1482.7,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10448\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":10,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/ComponentArray/(20, 128)",
            "value": 1498.8,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10448\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":10,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/Flux/(20, 128)",
            "value": 4862,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20736\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/SimpleChains/(20, 128)",
            "value": 1654.1,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":10,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 3, 128)",
            "value": 84249197.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=14310896\nallocs=97\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Zygote/(64, 64, 3, 128)",
            "value": 76847618,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22682688\nallocs=153\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Tracker/(64, 64, 3, 128)",
            "value": 155971883,
            "unit": "ns",
            "extra": "gctime=0\nmemory=70700488\nallocs=321\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/ReverseDiff/(64, 64, 3, 128)",
            "value": 171220777,
            "unit": "ns",
            "extra": "gctime=0\nmemory=46692208\nallocs=217\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Flux/(64, 64, 3, 128)",
            "value": 163576307,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28593648\nallocs=175\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/SimpleChains/(64, 64, 3, 128)",
            "value": 11997272,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5908032\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Enzyme/(64, 64, 3, 128)",
            "value": 189232749,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22683504\nallocs=173\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/NamedTuple/(64, 64, 3, 128)",
            "value": 15985673.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=7986720\nallocs=61\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/ComponentArray/(64, 64, 3, 128)",
            "value": 15990444,
            "unit": "ns",
            "extra": "gctime=0\nmemory=7986960\nallocs=60\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/Flux/(64, 64, 3, 128)",
            "value": 26907120,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13891072\nallocs=60\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/SimpleChains/(64, 64, 3, 128)",
            "value": 6388864,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 16)",
            "value": 1075154615,
            "unit": "ns",
            "extra": "gctime=0\nmemory=357822480\nallocs=3700\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 64)",
            "value": 2991995826,
            "unit": "ns",
            "extra": "gctime=58826195\nmemory=804519760\nallocs=3700\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 2)",
            "value": 203143942,
            "unit": "ns",
            "extra": "gctime=0\nmemory=227486784\nallocs=3412\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 16)",
            "value": 1327470294,
            "unit": "ns",
            "extra": "gctime=2708544\nmemory=699652992\nallocs=11148\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 64)",
            "value": 3714520054,
            "unit": "ns",
            "extra": "gctime=81324930\nmemory=1364599040\nallocs=11149\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 2)",
            "value": 393360752,
            "unit": "ns",
            "extra": "gctime=6642398\nmemory=505662672\nallocs=10848\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 16)",
            "value": 1513048001,
            "unit": "ns",
            "extra": "gctime=2528146\nmemory=430127552\nallocs=10918\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 64)",
            "value": 4465469598,
            "unit": "ns",
            "extra": "gctime=238339998\nmemory=1092309152\nallocs=10918\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 2)",
            "value": 427939919,
            "unit": "ns",
            "extra": "gctime=0\nmemory=236943008\nallocs=10618\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 16)",
            "value": 373761436,
            "unit": "ns",
            "extra": "gctime=0\nmemory=71415584\nallocs=1142\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 64)",
            "value": 904133788,
            "unit": "ns",
            "extra": "gctime=0\nmemory=185155264\nallocs=1142\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 2)",
            "value": 51595239.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=38216624\nallocs=1004\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 16)",
            "value": 374634761,
            "unit": "ns",
            "extra": "gctime=0\nmemory=71426944\nallocs=1269\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 64)",
            "value": 915540600,
            "unit": "ns",
            "extra": "gctime=0\nmemory=185166624\nallocs=1269\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 2)",
            "value": 50975700,
            "unit": "ns",
            "extra": "gctime=0\nmemory=38226736\nallocs=1131\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 16)",
            "value": 519866147,
            "unit": "ns",
            "extra": "gctime=0\nmemory=89650704\nallocs=1098\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 64)",
            "value": 1395993770,
            "unit": "ns",
            "extra": "gctime=0\nmemory=258049360\nallocs=1098\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 2)",
            "value": 166463654,
            "unit": "ns",
            "extra": "gctime=0\nmemory=40509712\nallocs=957\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 64, 128)",
            "value": 1256120343.5,
            "unit": "ns",
            "extra": "gctime=33419634.5\nmemory=305218144\nallocs=98\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Zygote/(64, 64, 64, 128)",
            "value": 1585667224.5,
            "unit": "ns",
            "extra": "gctime=1766051.5\nmemory=483723552\nallocs=154\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Tracker/(64, 64, 64, 128)",
            "value": 2358338032,
            "unit": "ns",
            "extra": "gctime=258236303\nmemory=1508223432\nallocs=322\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/ReverseDiff/(64, 64, 64, 128)",
            "value": 2629770018,
            "unit": "ns",
            "extra": "gctime=183899293\nmemory=995973856\nallocs=217\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Flux/(64, 64, 64, 128)",
            "value": 2299487877,
            "unit": "ns",
            "extra": "gctime=3975514.5\nmemory=609690320\nallocs=176\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Enzyme/(64, 64, 64, 128)",
            "value": 2202187002,
            "unit": "ns",
            "extra": "gctime=0\nmemory=483724144\nallocs=175\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/NamedTuple/(64, 64, 64, 128)",
            "value": 288765459,
            "unit": "ns",
            "extra": "gctime=0\nmemory=170249632\nallocs=61\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/ComponentArray/(64, 64, 64, 128)",
            "value": 287589792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=170249872\nallocs=60\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/Flux/(64, 64, 64, 128)",
            "value": 474169861.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=296209792\nallocs=60\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 1, 128)",
            "value": 11665649,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4777552\nallocs=97\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Zygote/(64, 64, 1, 128)",
            "value": 34793586,
            "unit": "ns",
            "extra": "gctime=0\nmemory=7571216\nallocs=153\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Tracker/(64, 64, 1, 128)",
            "value": 16498583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23582232\nallocs=319\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/ReverseDiff/(64, 64, 1, 128)",
            "value": 21107303,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15577344\nallocs=215\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Flux/(64, 64, 1, 128)",
            "value": 15300235,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9545920\nallocs=175\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/SimpleChains/(64, 64, 1, 128)",
            "value": 1163279.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1970896\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Enzyme/(64, 64, 1, 128)",
            "value": 36078505,
            "unit": "ns",
            "extra": "gctime=0\nmemory=7572016\nallocs=171\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/NamedTuple/(64, 64, 1, 128)",
            "value": 4714693.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2666656\nallocs=61\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/ComponentArray/(64, 64, 1, 128)",
            "value": 4687438,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2666896\nallocs=60\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/Flux/(64, 64, 1, 128)",
            "value": 1991858.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4634752\nallocs=60\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/SimpleChains/(64, 64, 1, 128)",
            "value": 198611,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/ReverseDiff (compiled)/(200, 128)",
            "value": 379980,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102672\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Zygote/(200, 128)",
            "value": 213118.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=470896\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Tracker/(200, 128)",
            "value": 388676,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1614552\nallocs=149\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/ReverseDiff/(200, 128)",
            "value": 512398,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1040224\nallocs=66\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Flux/(200, 128)",
            "value": 293543.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=571712\nallocs=33\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/SimpleChains/(200, 128)",
            "value": 407692,
            "unit": "ns",
            "extra": "gctime=0\nmemory=747872\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Enzyme/(200, 128)",
            "value": 428922,
            "unit": "ns",
            "extra": "gctime=0\nmemory=205408\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/NamedTuple/(200, 128)",
            "value": 57699,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102528\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/ComponentArray/(200, 128)",
            "value": 60272,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102528\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/Flux/(200, 128)",
            "value": 92693,
            "unit": "ns",
            "extra": "gctime=0\nmemory=204896\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/SimpleChains/(200, 128)",
            "value": 104605,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 16, 128)",
            "value": 336588187,
            "unit": "ns",
            "extra": "gctime=0\nmemory=76285088\nallocs=97\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Zygote/(64, 64, 16, 128)",
            "value": 286481480.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=120914848\nallocs=153\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Tracker/(64, 64, 16, 128)",
            "value": 553202812.5,
            "unit": "ns",
            "extra": "gctime=2479019\nmemory=376990536\nallocs=319\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/ReverseDiff/(64, 64, 16, 128)",
            "value": 631696024,
            "unit": "ns",
            "extra": "gctime=0\nmemory=248953248\nallocs=215\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Flux/(64, 64, 16, 128)",
            "value": 584369049,
            "unit": "ns",
            "extra": "gctime=0\nmemory=152411472\nallocs=175\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/SimpleChains/(64, 64, 16, 128)",
            "value": 332985122,
            "unit": "ns",
            "extra": "gctime=0\nmemory=31520784\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Enzyme/(64, 64, 16, 128)",
            "value": 608969909,
            "unit": "ns",
            "extra": "gctime=0\nmemory=120915632\nallocs=174\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/NamedTuple/(64, 64, 16, 128)",
            "value": 40188939,
            "unit": "ns",
            "extra": "gctime=0\nmemory=42567328\nallocs=61\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/ComponentArray/(64, 64, 16, 128)",
            "value": 40059867,
            "unit": "ns",
            "extra": "gctime=0\nmemory=42567568\nallocs=60\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/Flux/(64, 64, 16, 128)",
            "value": 109125318.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=74057344\nallocs=60\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/SimpleChains/(64, 64, 16, 128)",
            "value": 28648755,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/ReverseDiff (compiled)/(2000, 128)",
            "value": 21170151.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024272\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Zygote/(2000, 128)",
            "value": 21223422,
            "unit": "ns",
            "extra": "gctime=0\nmemory=19082928\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Tracker/(2000, 128)",
            "value": 22980720,
            "unit": "ns",
            "extra": "gctime=0\nmemory=59293848\nallocs=149\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/ReverseDiff/(2000, 128)",
            "value": 26450851,
            "unit": "ns",
            "extra": "gctime=0\nmemory=39178656\nallocs=66\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Flux/(2000, 128)",
            "value": 19326108.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20105344\nallocs=33\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Enzyme/(2000, 128)",
            "value": 20888920,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2048608\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/NamedTuple/(2000, 128)",
            "value": 6116570,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024128\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/ComponentArray/(2000, 128)",
            "value": 6384401,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024128\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/Flux/(2000, 128)",
            "value": 6570749,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2048096\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "avik.pal.2017@gmail.com",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "distinct": true,
          "id": "a0095bb14364b33870e5c225d7a9bdb95c83a87c",
          "message": "test: try separating the test Project files\n\n[skip docs]",
          "timestamp": "2024-08-03T17:10:07-07:00",
          "tree_id": "d6c37f8aceaef54d7cfc7c6c78e5d7db39ec4535",
          "url": "https://github.com/LuxDL/Lux.jl/commit/a0095bb14364b33870e5c225d7a9bdb95c83a87c"
        },
        "date": 1722745699373,
        "tool": "julia",
        "benches": [
          {
            "name": "Dense(2 => 2)/cpu/reverse/ReverseDiff (compiled)/(2, 128)",
            "value": 4210.375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Zygote/(2, 128)",
            "value": 7280.857142857143,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6016\nallocs=42\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Tracker/(2, 128)",
            "value": 21329.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17224\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/ReverseDiff/(2, 128)",
            "value": 9790.4,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9968\nallocs=59\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":5,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Flux/(2, 128)",
            "value": 9162.25,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5472\nallocs=28\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":4,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/SimpleChains/(2, 128)",
            "value": 4474.625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3120\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Enzyme/(2, 128)",
            "value": 4985.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2688\nallocs=6\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/NamedTuple/(2, 128)",
            "value": 1582,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1168\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":10,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/ComponentArray/(2, 128)",
            "value": 1464.8,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1168\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":10,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/Flux/(2, 128)",
            "value": 1802.0566037735848,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2176\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":53,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/SimpleChains/(2, 128)",
            "value": 179.46036161335186,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":719,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/ReverseDiff (compiled)/(20, 128)",
            "value": 17282,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10592\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Zygote/(20, 128)",
            "value": 13104,
            "unit": "ns",
            "extra": "gctime=0\nmemory=35664\nallocs=42\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Tracker/(20, 128)",
            "value": 37641,
            "unit": "ns",
            "extra": "gctime=0\nmemory=124696\nallocs=135\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/ReverseDiff/(20, 128)",
            "value": 29095,
            "unit": "ns",
            "extra": "gctime=0\nmemory=78432\nallocs=57\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Flux/(20, 128)",
            "value": 21600,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44400\nallocs=28\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/SimpleChains/(20, 128)",
            "value": 17072,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17632\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Enzyme/(20, 128)",
            "value": 25698,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21248\nallocs=6\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/NamedTuple/(20, 128)",
            "value": 2609.3333333333335,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10448\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":9,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/ComponentArray/(20, 128)",
            "value": 2913.222222222222,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10448\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":9,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/Flux/(20, 128)",
            "value": 4874.857142857143,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20736\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/SimpleChains/(20, 128)",
            "value": 1654.1,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":10,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 3, 128)",
            "value": 88901650,
            "unit": "ns",
            "extra": "gctime=0\nmemory=14310896\nallocs=97\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Zygote/(64, 64, 3, 128)",
            "value": 76591182,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22682688\nallocs=153\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Tracker/(64, 64, 3, 128)",
            "value": 145816642,
            "unit": "ns",
            "extra": "gctime=0\nmemory=70700488\nallocs=321\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/ReverseDiff/(64, 64, 3, 128)",
            "value": 174645570,
            "unit": "ns",
            "extra": "gctime=0\nmemory=46692208\nallocs=217\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Flux/(64, 64, 3, 128)",
            "value": 162467847,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28593648\nallocs=175\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/SimpleChains/(64, 64, 3, 128)",
            "value": 11901058,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5908032\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Enzyme/(64, 64, 3, 128)",
            "value": 195200307.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22683504\nallocs=173\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/NamedTuple/(64, 64, 3, 128)",
            "value": 15544763.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=7986720\nallocs=61\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/ComponentArray/(64, 64, 3, 128)",
            "value": 15554299,
            "unit": "ns",
            "extra": "gctime=0\nmemory=7986960\nallocs=60\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/Flux/(64, 64, 3, 128)",
            "value": 42589068.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13891072\nallocs=60\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/SimpleChains/(64, 64, 3, 128)",
            "value": 6385176,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 16)",
            "value": 1065642865.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=357822480\nallocs=3700\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 64)",
            "value": 2933656388,
            "unit": "ns",
            "extra": "gctime=62374870\nmemory=804519760\nallocs=3700\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 2)",
            "value": 169506774,
            "unit": "ns",
            "extra": "gctime=0\nmemory=227486784\nallocs=3412\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 16)",
            "value": 1395702878,
            "unit": "ns",
            "extra": "gctime=73089588\nmemory=699652992\nallocs=11148\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 64)",
            "value": 3742059235,
            "unit": "ns",
            "extra": "gctime=211153828\nmemory=1364596864\nallocs=11148\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 2)",
            "value": 384902400.5,
            "unit": "ns",
            "extra": "gctime=4007862.5\nmemory=505662672\nallocs=10848\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 16)",
            "value": 1673204957.5,
            "unit": "ns",
            "extra": "gctime=117254599\nmemory=430127552\nallocs=10918\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 64)",
            "value": 4683819068,
            "unit": "ns",
            "extra": "gctime=295151783\nmemory=1092309152\nallocs=10918\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 2)",
            "value": 459579475,
            "unit": "ns",
            "extra": "gctime=0\nmemory=236943008\nallocs=10618\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 16)",
            "value": 361855245,
            "unit": "ns",
            "extra": "gctime=0\nmemory=71415584\nallocs=1142\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 64)",
            "value": 884689737,
            "unit": "ns",
            "extra": "gctime=1058309.5\nmemory=185155264\nallocs=1142\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 2)",
            "value": 56182817,
            "unit": "ns",
            "extra": "gctime=0\nmemory=38216624\nallocs=1004\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 16)",
            "value": 364122724,
            "unit": "ns",
            "extra": "gctime=0\nmemory=71426944\nallocs=1269\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 64)",
            "value": 880222233.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=185166624\nallocs=1269\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 2)",
            "value": 55637947,
            "unit": "ns",
            "extra": "gctime=0\nmemory=38226736\nallocs=1131\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 16)",
            "value": 562801142,
            "unit": "ns",
            "extra": "gctime=0\nmemory=89650704\nallocs=1098\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 64)",
            "value": 1517562171,
            "unit": "ns",
            "extra": "gctime=0\nmemory=258049360\nallocs=1098\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 2)",
            "value": 171205515,
            "unit": "ns",
            "extra": "gctime=0\nmemory=40509712\nallocs=957\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 64, 128)",
            "value": 1254741222,
            "unit": "ns",
            "extra": "gctime=0\nmemory=305218144\nallocs=98\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Zygote/(64, 64, 64, 128)",
            "value": 1561957190.5,
            "unit": "ns",
            "extra": "gctime=1589138.5\nmemory=483723552\nallocs=154\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Tracker/(64, 64, 64, 128)",
            "value": 2274530908,
            "unit": "ns",
            "extra": "gctime=257984701\nmemory=1508223432\nallocs=322\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/ReverseDiff/(64, 64, 64, 128)",
            "value": 2467776572,
            "unit": "ns",
            "extra": "gctime=194994711\nmemory=995973856\nallocs=217\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Flux/(64, 64, 64, 128)",
            "value": 2236433184,
            "unit": "ns",
            "extra": "gctime=3041253.5\nmemory=609690320\nallocs=176\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Enzyme/(64, 64, 64, 128)",
            "value": 2124313017,
            "unit": "ns",
            "extra": "gctime=0\nmemory=483724144\nallocs=175\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/NamedTuple/(64, 64, 64, 128)",
            "value": 285138498,
            "unit": "ns",
            "extra": "gctime=0\nmemory=170249632\nallocs=61\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/ComponentArray/(64, 64, 64, 128)",
            "value": 284816604,
            "unit": "ns",
            "extra": "gctime=0\nmemory=170249872\nallocs=60\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/Flux/(64, 64, 64, 128)",
            "value": 426557087,
            "unit": "ns",
            "extra": "gctime=0\nmemory=296209792\nallocs=60\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 1, 128)",
            "value": 11831175,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4777552\nallocs=97\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Zygote/(64, 64, 1, 128)",
            "value": 34523223,
            "unit": "ns",
            "extra": "gctime=0\nmemory=7571216\nallocs=153\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Tracker/(64, 64, 1, 128)",
            "value": 16447087,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23582232\nallocs=319\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/ReverseDiff/(64, 64, 1, 128)",
            "value": 21042971,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15577344\nallocs=215\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Flux/(64, 64, 1, 128)",
            "value": 15151300,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9545920\nallocs=175\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/SimpleChains/(64, 64, 1, 128)",
            "value": 1151983,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1970896\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Enzyme/(64, 64, 1, 128)",
            "value": 35724522,
            "unit": "ns",
            "extra": "gctime=0\nmemory=7572016\nallocs=171\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/NamedTuple/(64, 64, 1, 128)",
            "value": 4546295,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2666656\nallocs=61\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/ComponentArray/(64, 64, 1, 128)",
            "value": 4524386,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2666896\nallocs=60\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/Flux/(64, 64, 1, 128)",
            "value": 1970497,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4634752\nallocs=60\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/SimpleChains/(64, 64, 1, 128)",
            "value": 197294,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/ReverseDiff (compiled)/(200, 128)",
            "value": 380051,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102672\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Zygote/(200, 128)",
            "value": 222526,
            "unit": "ns",
            "extra": "gctime=0\nmemory=470896\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Tracker/(200, 128)",
            "value": 384974.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1614552\nallocs=149\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/ReverseDiff/(200, 128)",
            "value": 528056,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1040224\nallocs=66\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Flux/(200, 128)",
            "value": 294290,
            "unit": "ns",
            "extra": "gctime=0\nmemory=571712\nallocs=33\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/SimpleChains/(200, 128)",
            "value": 408554,
            "unit": "ns",
            "extra": "gctime=0\nmemory=747872\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Enzyme/(200, 128)",
            "value": 429642.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=205408\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/NamedTuple/(200, 128)",
            "value": 70512,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102528\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/ComponentArray/(200, 128)",
            "value": 70061,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102528\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/Flux/(200, 128)",
            "value": 92674,
            "unit": "ns",
            "extra": "gctime=0\nmemory=204896\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/SimpleChains/(200, 128)",
            "value": 104425,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 16, 128)",
            "value": 305601445,
            "unit": "ns",
            "extra": "gctime=0\nmemory=76285088\nallocs=97\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Zygote/(64, 64, 16, 128)",
            "value": 288797620.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=120914848\nallocs=153\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Tracker/(64, 64, 16, 128)",
            "value": 544934785.5,
            "unit": "ns",
            "extra": "gctime=2135149\nmemory=376990536\nallocs=319\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/ReverseDiff/(64, 64, 16, 128)",
            "value": 589046818,
            "unit": "ns",
            "extra": "gctime=0\nmemory=248953248\nallocs=215\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Flux/(64, 64, 16, 128)",
            "value": 552663074,
            "unit": "ns",
            "extra": "gctime=0\nmemory=152411472\nallocs=175\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/SimpleChains/(64, 64, 16, 128)",
            "value": 319651032.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=31520784\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Enzyme/(64, 64, 16, 128)",
            "value": 660000513,
            "unit": "ns",
            "extra": "gctime=0\nmemory=120915632\nallocs=174\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/NamedTuple/(64, 64, 16, 128)",
            "value": 39489513.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=42567328\nallocs=61\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/ComponentArray/(64, 64, 16, 128)",
            "value": 39313659,
            "unit": "ns",
            "extra": "gctime=0\nmemory=42567568\nallocs=60\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/Flux/(64, 64, 16, 128)",
            "value": 99796304,
            "unit": "ns",
            "extra": "gctime=0\nmemory=74057344\nallocs=60\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/SimpleChains/(64, 64, 16, 128)",
            "value": 28394041,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/ReverseDiff (compiled)/(2000, 128)",
            "value": 21157343.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024272\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Zygote/(2000, 128)",
            "value": 16678967,
            "unit": "ns",
            "extra": "gctime=0\nmemory=19082928\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Tracker/(2000, 128)",
            "value": 22876715,
            "unit": "ns",
            "extra": "gctime=0\nmemory=59293848\nallocs=149\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/ReverseDiff/(2000, 128)",
            "value": 28080465,
            "unit": "ns",
            "extra": "gctime=0\nmemory=39178656\nallocs=66\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Flux/(2000, 128)",
            "value": 19385408,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20105344\nallocs=33\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Enzyme/(2000, 128)",
            "value": 20728329,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2048608\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/NamedTuple/(2000, 128)",
            "value": 5545671.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024128\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/ComponentArray/(2000, 128)",
            "value": 5626649,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024128\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/Flux/(2000, 128)",
            "value": 6521885,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2048096\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "avik.pal.2017@gmail.com",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "distinct": true,
          "id": "b6171a6d2ce4b30e2ebaa2526e0fce5832f5a193",
          "message": "fix: relax test tolerances",
          "timestamp": "2024-08-04T16:02:33-07:00",
          "tree_id": "a949e67a9ddf47f0e6b63cdc9107b78ecf5b86a4",
          "url": "https://github.com/LuxDL/Lux.jl/commit/b6171a6d2ce4b30e2ebaa2526e0fce5832f5a193"
        },
        "date": 1722817695493,
        "tool": "julia",
        "benches": [
          {
            "name": "Dense(2 => 2)/cpu/reverse/ReverseDiff (compiled)/(2, 128)",
            "value": 3675.625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Zygote/(2, 128)",
            "value": 8093.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6208\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Tracker/(2, 128)",
            "value": 21210,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17224\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/ReverseDiff/(2, 128)",
            "value": 9748.2,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9968\nallocs=59\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":5,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Flux/(2, 128)",
            "value": 9167.2,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5472\nallocs=28\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":5,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/SimpleChains/(2, 128)",
            "value": 4470.875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3120\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/reverse/Enzyme/(2, 128)",
            "value": 4956.875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2688\nallocs=6\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/NamedTuple/(2, 128)",
            "value": 2373.4,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=7\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":10,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/ComponentArray/(2, 128)",
            "value": 2270.3,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":10,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/Flux/(2, 128)",
            "value": 1790.017543859649,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2176\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":57,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2 => 2)/cpu/forward/SimpleChains/(2, 128)",
            "value": 179.70239774330042,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":709,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/ReverseDiff (compiled)/(20, 128)",
            "value": 17562.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10592\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Zygote/(20, 128)",
            "value": 24787,
            "unit": "ns",
            "extra": "gctime=0\nmemory=35856\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Tracker/(20, 128)",
            "value": 38393,
            "unit": "ns",
            "extra": "gctime=0\nmemory=124696\nallocs=135\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/ReverseDiff/(20, 128)",
            "value": 29025,
            "unit": "ns",
            "extra": "gctime=0\nmemory=78432\nallocs=57\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Flux/(20, 128)",
            "value": 21590,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44400\nallocs=28\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/SimpleChains/(20, 128)",
            "value": 17092,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17632\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/reverse/Enzyme/(20, 128)",
            "value": 25648,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21248\nallocs=6\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/NamedTuple/(20, 128)",
            "value": 20248,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10640\nallocs=7\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/ComponentArray/(20, 128)",
            "value": 14448,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10640\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/Flux/(20, 128)",
            "value": 4846.285714285715,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20736\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(20 => 20)/cpu/forward/SimpleChains/(20, 128)",
            "value": 1659.2,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":10,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 3, 128)",
            "value": 77690170,
            "unit": "ns",
            "extra": "gctime=0\nmemory=14310896\nallocs=97\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Zygote/(64, 64, 3, 128)",
            "value": 76782338,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22682688\nallocs=153\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Tracker/(64, 64, 3, 128)",
            "value": 155414925,
            "unit": "ns",
            "extra": "gctime=0\nmemory=70700488\nallocs=321\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/ReverseDiff/(64, 64, 3, 128)",
            "value": 167638289.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=46692208\nallocs=217\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Flux/(64, 64, 3, 128)",
            "value": 142842293.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28593648\nallocs=175\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/SimpleChains/(64, 64, 3, 128)",
            "value": 11557321.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5908032\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/reverse/Enzyme/(64, 64, 3, 128)",
            "value": 199234044.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22683504\nallocs=173\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/NamedTuple/(64, 64, 3, 128)",
            "value": 15528408.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=7986720\nallocs=61\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/ComponentArray/(64, 64, 3, 128)",
            "value": 15540189,
            "unit": "ns",
            "extra": "gctime=0\nmemory=7986960\nallocs=60\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/Flux/(64, 64, 3, 128)",
            "value": 30661456,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13891072\nallocs=60\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 3 => 3)/cpu/forward/SimpleChains/(64, 64, 3, 128)",
            "value": 6376663,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 16)",
            "value": 1064055959.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=357827904\nallocs=3802\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 64)",
            "value": 2970205700,
            "unit": "ns",
            "extra": "gctime=72266467\nmemory=804525184\nallocs=3802\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Zygote/(32, 32, 3, 2)",
            "value": 178121161,
            "unit": "ns",
            "extra": "gctime=0\nmemory=227492208\nallocs=3514\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 16)",
            "value": 1320655778,
            "unit": "ns",
            "extra": "gctime=14806767\nmemory=699655168\nallocs=11149\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 64)",
            "value": 3516351096,
            "unit": "ns",
            "extra": "gctime=89330912\nmemory=1364596864\nallocs=11148\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Tracker/(32, 32, 3, 2)",
            "value": 344809509,
            "unit": "ns",
            "extra": "gctime=3291601\nmemory=505662672\nallocs=10848\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 16)",
            "value": 1431616033,
            "unit": "ns",
            "extra": "gctime=0\nmemory=430127552\nallocs=10918\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 64)",
            "value": 4058579611,
            "unit": "ns",
            "extra": "gctime=95430923\nmemory=1092309152\nallocs=10918\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/reverse/Flux/(32, 32, 3, 2)",
            "value": 436008182,
            "unit": "ns",
            "extra": "gctime=0\nmemory=236943008\nallocs=10618\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 16)",
            "value": 381866129,
            "unit": "ns",
            "extra": "gctime=0\nmemory=71416160\nallocs=1154\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 64)",
            "value": 905256978,
            "unit": "ns",
            "extra": "gctime=0\nmemory=185155840\nallocs=1154\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/NamedTuple/(32, 32, 3, 2)",
            "value": 54567006.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=38217200\nallocs=1016\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 16)",
            "value": 382293897,
            "unit": "ns",
            "extra": "gctime=0\nmemory=71427520\nallocs=1281\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 64)",
            "value": 870357323.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=185167200\nallocs=1281\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/ComponentArray/(32, 32, 3, 2)",
            "value": 54472914.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=38227312\nallocs=1143\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 16)",
            "value": 551222188,
            "unit": "ns",
            "extra": "gctime=0\nmemory=89650704\nallocs=1098\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 64)",
            "value": 1387168504,
            "unit": "ns",
            "extra": "gctime=0\nmemory=258049360\nallocs=1098\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "vgg16/cpu/forward/Flux/(32, 32, 3, 2)",
            "value": 164122645,
            "unit": "ns",
            "extra": "gctime=0\nmemory=40509712\nallocs=957\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 64, 128)",
            "value": 1180058919,
            "unit": "ns",
            "extra": "gctime=0\nmemory=305218144\nallocs=98\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Zygote/(64, 64, 64, 128)",
            "value": 1610297742,
            "unit": "ns",
            "extra": "gctime=0\nmemory=483723552\nallocs=154\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Tracker/(64, 64, 64, 128)",
            "value": 2289727615.5,
            "unit": "ns",
            "extra": "gctime=244472253\nmemory=1508223432\nallocs=322\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/ReverseDiff/(64, 64, 64, 128)",
            "value": 2640437136,
            "unit": "ns",
            "extra": "gctime=178349468\nmemory=995973856\nallocs=217\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Flux/(64, 64, 64, 128)",
            "value": 2193753011.5,
            "unit": "ns",
            "extra": "gctime=2837414\nmemory=609690320\nallocs=176\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/reverse/Enzyme/(64, 64, 64, 128)",
            "value": 2122924359,
            "unit": "ns",
            "extra": "gctime=0\nmemory=483724144\nallocs=175\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/NamedTuple/(64, 64, 64, 128)",
            "value": 282003619,
            "unit": "ns",
            "extra": "gctime=0\nmemory=170249632\nallocs=61\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/ComponentArray/(64, 64, 64, 128)",
            "value": 286261947,
            "unit": "ns",
            "extra": "gctime=0\nmemory=170249872\nallocs=60\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 64 => 64)/cpu/forward/Flux/(64, 64, 64, 128)",
            "value": 437257287,
            "unit": "ns",
            "extra": "gctime=0\nmemory=296209792\nallocs=60\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 1, 128)",
            "value": 11806435,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4777552\nallocs=97\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Zygote/(64, 64, 1, 128)",
            "value": 34527638,
            "unit": "ns",
            "extra": "gctime=0\nmemory=7571216\nallocs=153\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Tracker/(64, 64, 1, 128)",
            "value": 16364743,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23582232\nallocs=319\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/ReverseDiff/(64, 64, 1, 128)",
            "value": 21004093,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15577344\nallocs=215\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Flux/(64, 64, 1, 128)",
            "value": 15284140,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9545920\nallocs=175\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/SimpleChains/(64, 64, 1, 128)",
            "value": 1148921.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1970896\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/reverse/Enzyme/(64, 64, 1, 128)",
            "value": 35777843.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=7572016\nallocs=171\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/NamedTuple/(64, 64, 1, 128)",
            "value": 4500694,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2666656\nallocs=61\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/ComponentArray/(64, 64, 1, 128)",
            "value": 4506207,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2666896\nallocs=60\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/Flux/(64, 64, 1, 128)",
            "value": 2045686,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4634752\nallocs=60\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 1 => 1)/cpu/forward/SimpleChains/(64, 64, 1, 128)",
            "value": 196300,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/ReverseDiff (compiled)/(200, 128)",
            "value": 378068,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102672\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Zygote/(200, 128)",
            "value": 314462,
            "unit": "ns",
            "extra": "gctime=0\nmemory=471088\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Tracker/(200, 128)",
            "value": 377972,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1614552\nallocs=149\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/ReverseDiff/(200, 128)",
            "value": 520691,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1040224\nallocs=66\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Flux/(200, 128)",
            "value": 289716,
            "unit": "ns",
            "extra": "gctime=0\nmemory=571712\nallocs=33\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/SimpleChains/(200, 128)",
            "value": 401777,
            "unit": "ns",
            "extra": "gctime=0\nmemory=747872\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/reverse/Enzyme/(200, 128)",
            "value": 425321,
            "unit": "ns",
            "extra": "gctime=0\nmemory=205408\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/NamedTuple/(200, 128)",
            "value": 157406,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102720\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/ComponentArray/(200, 128)",
            "value": 162456,
            "unit": "ns",
            "extra": "gctime=0\nmemory=102720\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/Flux/(200, 128)",
            "value": 91953,
            "unit": "ns",
            "extra": "gctime=0\nmemory=204896\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(200 => 200)/cpu/forward/SimpleChains/(200, 128)",
            "value": 104407,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/ReverseDiff (compiled)/(64, 64, 16, 128)",
            "value": 297649242,
            "unit": "ns",
            "extra": "gctime=0\nmemory=76285088\nallocs=97\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Zygote/(64, 64, 16, 128)",
            "value": 287837994,
            "unit": "ns",
            "extra": "gctime=0\nmemory=120914848\nallocs=153\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Tracker/(64, 64, 16, 128)",
            "value": 545531151.5,
            "unit": "ns",
            "extra": "gctime=2681547\nmemory=376990536\nallocs=319\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/ReverseDiff/(64, 64, 16, 128)",
            "value": 655809148,
            "unit": "ns",
            "extra": "gctime=0\nmemory=248953248\nallocs=215\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Flux/(64, 64, 16, 128)",
            "value": 554893727,
            "unit": "ns",
            "extra": "gctime=0\nmemory=152411472\nallocs=175\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/SimpleChains/(64, 64, 16, 128)",
            "value": 316084028.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=31520784\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/reverse/Enzyme/(64, 64, 16, 128)",
            "value": 583442251.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=120915632\nallocs=174\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/NamedTuple/(64, 64, 16, 128)",
            "value": 40159465,
            "unit": "ns",
            "extra": "gctime=0\nmemory=42567328\nallocs=61\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/ComponentArray/(64, 64, 16, 128)",
            "value": 40173961.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=42567568\nallocs=60\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/Flux/(64, 64, 16, 128)",
            "value": 96663497,
            "unit": "ns",
            "extra": "gctime=0\nmemory=74057344\nallocs=60\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Conv((3, 3), 16 => 16)/cpu/forward/SimpleChains/(64, 64, 16, 128)",
            "value": 28321531,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/ReverseDiff (compiled)/(2000, 128)",
            "value": 21078472,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024272\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Zygote/(2000, 128)",
            "value": 17393481,
            "unit": "ns",
            "extra": "gctime=0\nmemory=19083120\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Tracker/(2000, 128)",
            "value": 22657728,
            "unit": "ns",
            "extra": "gctime=0\nmemory=59293848\nallocs=149\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/ReverseDiff/(2000, 128)",
            "value": 28019412,
            "unit": "ns",
            "extra": "gctime=0\nmemory=39178656\nallocs=66\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Flux/(2000, 128)",
            "value": 19298592.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20105344\nallocs=33\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/reverse/Enzyme/(2000, 128)",
            "value": 20720819,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2048608\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/NamedTuple/(2000, 128)",
            "value": 6086608,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024320\nallocs=8\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/ComponentArray/(2000, 128)",
            "value": 6101998,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1024320\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "Dense(2000 => 2000)/cpu/forward/Flux/(2000, 128)",
            "value": 6509879.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2048096\nallocs=4\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      }
    ]
  }
}