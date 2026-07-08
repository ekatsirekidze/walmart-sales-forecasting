# Walmart Recruiting — Store Sales Forecasting

გუნდური საბოლოო პროექტი: 45 მაღაზიის დეპარტამენტების ყოველკვირეული გაყიდვების პროგნოზირება.

- **Kaggle:** https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting
- **MLflow (DagsHub):** https://dagshub.com/ekatsirekidze/walmart-sales-forecasting.mlflow
- **გუნდი:** `<სახელი 1>` (PatchTST, DLinear, LightGBM, ARIMA/SARIMA), `<სახელი 2>` (N-BEATS, XGBoost, Prophet, TimesFM)

## 1. ამოცანა და მეტრიკა

მოცემულია 3331 (Store, Dept) დროითი მწკრივი, 143 კვირა (2010-02 — 2012-10). საპროგნოზოა **39 კვირა** (2012-11 — 2013-07), რომელიც მოიცავს Thanksgiving-ს, Christmas-ს და Super Bowl-ს.

მეტრიკა — **WMAE**: სადღესასწაულო კვირებს აქვთ **5-მაგი წონა**. ამის გამო:
- ვწვრთნით `sample_weight = 1 + 4·IsHoliday`-თი (ან L1/MAE ლოსით) — ვაოპტიმიზებთ ზუსტად იმას, რასაც Kaggle ზომავს;
- ტარგეტს არ ვალოგარითმებთ — მეტრიკა ნედლ დოლარებზეა.

## 2. EDA — მთავარი დასკვნები

- ძლიერი წლიური სეზონურობა (lag-52 ავტოკორელაციის პიკი) — ეს განსაზღვრავს ყველა მოდელის დიზაინს.
- გაყიდვების პიკი Thanksgiving/Christmas-ის **წინა** კვირებშია; თავად საშობაო კვირაში ვარდნაა.
- MarkDown1-5 მხოლოდ 2011 ნოემბრიდან არსებობს (ტესტის პერიოდში — სრულად). NA = აქცია არ მიმდინარეობს → 0 + `_present` ფლაგი.
- CPI/Unemployment აკლია ტესტის ბოლო კვირებში → forward-fill მაღაზიის მიხედვით.
- 1285 უარყოფითი გაყიდვა (დაბრუნებები) — ვტოვებთ.
- ტესტში 11 სერიაა, რომელსაც ისტორია საერთოდ არ აქვს (cold-start).

`<აქ ჩასვით EDA-ს გრაფიკები>`

## 3. ვალიდაციის სტრატეგია

Expanding-window (rolling-origin) სქემა — ორივე fold-ი ტესტის ზუსტი ფორმისაა (39 კვირა):

| Fold | Train | Validation | შენიშვნა |
|---|---|---|---|
| 1 (მთავარი) | ≤ 2011-10-28 | 2011-11-04 → 2012-07-27 | შეიცავს Thanksgiving/Christmas/Super Bowl-ს, როგორც ტესტი |
| 2 (დამხმარე) | ≤ 2012-01-27 | 2012-02-03 → 2012-10-26 | ჩვეულებრივი კვირების sanity-check |

გადაწყვეტილებები მიიღება Fold 1-ზე. K-fold shuffle აქ დაუშვებელია — მომავლის ინფორმაცია გაჟონავდა წარსულში (leakage).

## 4. არქიტექტურები და შედეგები

თითო არქიტექტურას აქვს ცალკე ნოუთბუქი (`model_experiment_*.ipynb`) და ცალკე MLflow ექსპერიმენტი (`*_Training`) შესაბამისი run-ებით (Cleaning, Feature_Selection, CV, Final).

### 4.1 Baseline — Seasonal Naive (lag-52)
პროგნოზი = იგივე სერიის გაყიდვები 52 კვირის წინ (fallback: lag 53/51, სერიის მედიანა, დეპარტამენტის მედიანა).

- Fold-1 WMAE **2031** (holiday MAE 2300 / non-holiday 1918), Fold-2 **1803**
- Kaggle: public **2945.64** / private **3027.85**
- დანიშნულება: (1) ხარისხის ზღვარი — ამაზე უარესი მოდელი გაფუჭებულია; (2) ლოკალური ვალიდაციისა და leaderboard-ის კალიბრაცია (LB ≈ 1.3-1.45 × Fold-1).

### 4.2 LightGBM
გლობალური მოდელი ყველა სერიაზე; ყველა lag-ფიჩერი ≥ 39 კვირაა → **direct multi-horizon** პროგნოზი რეკურსიის გარეშე. წვრთნა `objective="l1"` + `sample_weight = 1+4·IsHoliday`.

**Round 1 (3 კონფიგი, Fold 1):**

| კონფიგი | WMAE | holiday MAE | non-holiday MAE |
|---|---|---|---|
| 127 leaves, lr 0.05, 800 trees | **1924.8** | 2355.1 | 1743.0 |
| 255 leaves, lr 0.03, 1500 trees | 1978.8 | 2540.4 | 1741.5 |
| 63 leaves, lr 0.05, 800 trees | 1943.0 | 2412.1 | 1744.8 |

დასკვნები: (1) დიდი ხეები (255 leaves) holiday-კვირებს გადაისწავლის — non-holiday შეცდომა იდენტურია, holiday კი +185; (2) მოდელი holiday-კვირებზე (MAE 2355) naive baseline-ზეც (2300) უარესი იყო → round 2-ის მთავარი სამიზნე.

Kaggle (round 1 + Christmas shift): public **2507.37** / private **2567.09**.

**Round 2 (holiday-კვირების გაუმჯობესება):**
- *Holiday-aligned lags* (lag-ი მოსწორებული დღესასწაულის თარიღზე და არა ფიქსირებულ 52 კვირაზე) — **უარყოფითი შედეგი**: holiday MAE 2355→2464. ერთი მოსწორებული წელი ზედმეტად ხმაურიანია და მოდელი მას ბრმად ენდობა. დოკუმენტირებულია MLflow-ში (`LightGBM_HolidayLags_Ablation`), ფიჩერი გამორთულია.
- *Naive blend holiday-კვირებზე*: დიდი-4 დღესასწაულის კვირებზე პროგნოზი = (1−w)·LightGBM + w·naive-lag52. Fold 1-ზე საუკეთესო w=0.6 → **WMAE 1866.5, holiday MAE 2158.9** — პირველად ვჯობნით naive-ს holiday-კვირებზეც.

მოდელი ინახება Registry-ში (`walmart-lightgbm`), **cloudpickle** ფორმატით — MLflow-ის default skops ფორმატი custom preprocessor-ს ვერ ასერიალიზებს.

### 4.3 DLinear
ტრენდი+სეზონურობის დეკომპოზიცია და თითო ხაზოვანი შრე — „საჭიროა კი ტრანსფორმერები პროგნოზირებისთვის?" კვლევის პასუხი. `<შედეგები>`

### 4.4 PatchTST
Patch-ტოკენიზაცია + channel-independent ტრანსფორმერი. `<შედეგები>`

### 4.5 ARIMA / SARIMA (თეორია + დემო)
`<ACF/PACF, სტაციონარულობა, რატომ ვერ იჭერს ARIMA წლიურ სეზონურობას და რას ამატებს SARIMA (P,D,Q,s); რატომ არ მასშტაბირდება 3331 სერიაზე>`

### 4.6 XGBoost / N-BEATS / Prophet / TimesFM (მეორე წევრი)
`<შედეგები>`

### შედარების ცხრილი

| მოდელი | Fold-1 WMAE | Kaggle Public | Kaggle Private |
|---|---|---|---|
| Seasonal Naive | 2031 | 2945.64 | 3027.85 |
| LightGBM (r1 + shift) | 1925 | 2507.37 | 2567.09 |
| LightGBM (r2 + shift + blend w=0.6) | 1866 | | |
| DLinear | | | |
| PatchTST | | | |
| SARIMA | | | |
| XGBoost | | | |
| N-BEATS | | | |
| Prophet | | | |
| Ensemble | | | |

## 5. Post-processing — „საშობაო წანაცვლება"

სატრენინგო წლებში შობის მომდევნო კვირა შეიცავდა 0 (2010) ან 1 (2011) წინასაშობაო დღეს; სატესტო 2012-12-28 კვირა კი — 3-ს (22-24 დეკემბერი). ისტორიაზე ნაწვრთნი ნებისმიერი მოდელი ამ 5-წონიან კვირას აუფასურებს. შესწორება: `pred_52 += (2.5/7)·(pred_51 − pred_52)`.

**Post-processing ablation (LightGBM r2 მოდელი, სამი submission):**

| ვარიანტი | Kaggle Public | Kaggle Private |
|---|---|---|
| post-processing-ის გარეშე | 2765.77 | 2845.19 |
| + Christmas shift | `<public>` | 2591.81 |
| + shift + blend ოთხივე დღესასწაულზე (w=0.5) | 2604.08 | 2671.29 |

ორი გაკვეთილი leaderboard-იდან:
1. **Christmas shift-ი ~250 ქულა ღირს** (2845→2592 private) — ერთი კალენდრული დაკვირვება მთელ ჰიპერპარამეტრების ტუნინგზე მეტს იძლევა.
2. **Blend-მა საშობაო კვირაზე shift-ის ეფექტი გააუქმა** (2592→2671): naive lag-52 შარშანდელ პოსტ-საშობაო კვირას იღებს, სადაც წინასაშობაო დღეების რაოდენობა წლიდან წლამდე იცვლება (0/1/3 დღე). გამოსავალი (round 3): blend მხოლოდ Super Bowl / Labor Day / Thanksgiving კვირებზე — Fold 1-ზეც უკეთესია (holiday MAE 2159→2102) და shift-საც აღარ ეწინააღმდეგება.
3. კონფიგების შერჩევის ხმაური: Fold 1-მა config 1 არჩია (1901 vs 1925), leaderboard-მა კი config 0 (2567 vs 2592 private) — ~25-ქულიანი სხვაობა ორივე მიმართულებით = ერთ fold-ზე შერჩევის ცდომილება. საბოლოოდ ორივე კონფიგი გაიგზავნა და შეირჩა ტესტის მტკიცებულებით.

## 6. დასკვნები

`<რომელმა არქიტექტურამ მოიგო და რატომ: გლობალური vs ლოკალური მოდელები; direct vs recursive პროგნოზი; რატომ ჯობნის ხეებზე დაფუძნებული მოდელი კარგი ფიჩერებით ტრანსფორმერებს მცირე, ძლიერად სეზონურ მონაცემებზე; foundation მოდელების შედეგი>`

## 7. რეპოზიტორიის სტრუქტურა და გაშვება

```
src/            საერთო კოდი: მეტრიკა, ვალიდაცია, პრეპროცესინგი, post-processing
eda.ipynb       ერთობლივი EDA
model_experiment_<Arch>.ipynb   თითო არქიტექტურა
model_inference.ipynb           საუკეთესო მოდელი Model Registry-დან → submission
```

```bash
python -m venv .venv && .venv/Scripts/activate   # Windows
pip install -r requirements.txt
# ჩაწერეთ Kaggle-ის CSV-ები data/ საქაღალდეში
# MLflow (DagsHub) წვდომა: MLFLOW_TRACKING_URI / USERNAME / PASSWORD ცვლადები
jupyter lab
```

საბოლოო მოდელი ინახება MLflow Model Registry-ში სახელით **`walmart-best-model`**; `model_inference.ipynb` მას პირდაპირ რეესტრიდან ტვირთავს და ნედლ ტესტზე უშვებს.
