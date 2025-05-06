# 👟 SoleSelect – Content-Based Shoe Recommendation System

**SoleSelect** is a machine learning project that recommends similar shoes based on product descriptions using TF-IDF vectorization and cosine similarity. It’s tailored for e-commerce datasets like Flipkart or Nike catalogs.

---

## 🚀 Features

- ✅ Content-based filtering (no user data required)
- ✅ TF-IDF vectorization of shoe descriptions
- ✅ Cosine similarity scoring
- ✅ Top-N recommendations per shoe
- ✅ Visualization of product similarity matrix
- ✅ Accuracy estimation using average top-k similarity

---

## 🗂️ Project Structure
SoleSelect/
 
├── code---soleselect.py # Equivalent Python script
├── data/
│ └── shoe_data.csv # Shoe dataset (Flipkart/Nike style)
├── output/
│ ├── sam_recommendations.csv # Example output
│ ├── similarity_heatmap.png # Visualized cosine similarity
│ 
└── README.md # You're reading it!


---

## 📈 Example Output

### 🔍 Top-5 Recommendations:

Nike Air Zoom Pegasus 39
Nike Revolution 6
Nike Winflo 8
Nike Flex Experience Run 10
Nike Downshifter 11


## 📊 Dataset Source

The dataset is a curated compilation of shoe product listings sourced from public e-commerce platforms such as Flipkart via the [Crawlfeed API](https://www.crawlfeed.com/). It has been standardized and cleaned for use in product recommendation tasks.
