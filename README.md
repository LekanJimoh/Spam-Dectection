# Spam-Dectection
### **README: Spam Detection Using Machine Learning**

---

## **Project Overview**
This project focuses on building a machine learning model to classify messages as either **Spam** (unwanted) or **Ham** (legitimate). The model uses natural language processing (NLP) techniques to preprocess and analyze text data, ensuring accurate classification and the ability to detect spam messages in real-world applications like email or SMS filtering.

---

## **Key Features**
- **End-to-End Pipeline:** From data cleaning and exploration to model training and evaluation.
- **Text Preprocessing:** Includes tokenization, stop word removal, and lemmatization for effective feature extraction.
- **Machine Learning Models:** Comparison of various classifiers like Random Forest, Naive Bayes, Logistic Regression, and more.
- **Deployment-Ready:** Final model is optimized for real-time or large-scale spam detection.

---

## **Technologies Used**
- **Programming Language:** Python  
- **Libraries:**  
  - Data Manipulation: `Pandas`, `NumPy`  
  - Visualization: `Matplotlib`, `Seaborn`, `WordCloud`  
  - NLP: `NLTK`  
  - Machine Learning: `Scikit-learn`  
- **Deployment Tool:** `Streamlit`  

---

## **Project Workflow**
1. **Data Cleaning and Preparation**
   - Checked for missing and duplicate values.
   - Removed duplicates to ensure data quality.
   - Tokenized messages and applied lemmatization.

2. **Exploratory Data Analysis (EDA)**
   - Generated word clouds to visualize common words in spam and ham messages.
   - Analyzed class distribution to address class imbalance.

3. **Model Training and Testing**
   - Split data into training and testing sets (80/20).
   - Trained and compared various models for classification.
   - Evaluated models using metrics such as accuracy, precision, recall, and F1-score.

4. **Final Model Selection**
   - Selected **Random Forest Classifier** for its superior performance:
     - **Accuracy:** 97.8%
     - **Precision:** 95.6%
     - **Recall:** 94.5%
     - **F1-Score:** 95.0%

5. **Deployment**
   - Built an interactive app using Streamlit for real-time spam detection.

---

## **How to Use This Repository**
1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/spam-detection.git
   cd spam-detection
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App**
   ```bash
   streamlit run app.py
   ```

4. **Test the Model**
   - Input sample messages in the app to classify them as Spam or Ham.

---

## **Dataset**
- The dataset contains labeled messages, with `Label` as Spam or Ham and `Message` as the text.
- Duplicate entries were removed, resulting in a clean dataset of **5,144 unique messages**.

---

## **Results**
- The trained model accurately classifies messages and provides robust spam detection.
- High-performance metrics ensure reliability for deployment in communication systems.

---

## **Contributing**
Contributions are welcome! Feel free to:
- Open issues for bug fixes or feature requests.
- Fork the repository and create pull requests for improvements.

---

## **License**
This project is licensed under the [MIT License](LICENSE).

---

Feel free to customize the `GitHub URL` and add any personal notes to the `Contributing` section!
