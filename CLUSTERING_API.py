from flask import Flask, request, jsonify, send_file
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, MeanShift, AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, fcluster
import os

app = Flask(__name__)

class KMeansClusterer:
    def init(self, data, n_clusters):
        """
        Initialize KMeans clustering.
        مقداردهی اولیه خوشهبندی KMeans.
        تهيئة KMeans للتصنيف.
        Initialiser le clustering KMeans.
        """
        self.data = data
        self.n_clusters = n_clusters
        self.model = None

    def fit_predict(self):
        """
        Apply KMeans clustering and return labels.
        اجرای خوشهبندی KMeans و بازگرداندن برچسبها.
        تطبيق KMeans وإرجاع التسميات.
        Appliquer KMeans et retourner les étiquettes.
        """
        self.model = KMeans(n_clusters=self.n_clusters)
        return self.model.fit_predict(self.data)


class DBSCANClusterer:
    def init(self, data, eps=0.5, min_samples=5):
        """
        Initialize DBSCAN clustering.
        مقداردهی اولیه خوشهبندی DBSCAN.
        تهيئة DBSCAN للتصنيف.
        Initialiser le clustering DBSCAN.
        """
        self.data = data
        self.eps = eps
        self.min_samples = min_samples
        self.model = None

    def fit_predict(self):
        """
        Apply DBSCAN clustering and return labels.
        اجرای خوشهبندی DBSCAN و بازگرداندن برچسبها.
        تطبيق DBSCAN وإرجاع التسميات.
        Appliquer DBSCAN et retourner les étiquettes.
        """
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        return self.model.fit_predict(self.data)


class HierarchicalClusterer:
    def init(self, data, method='ward', threshold=1.5, criterion='distance'):
        """
        Initialize Hierarchical clustering.
        مقداردهی اولیه خوشهبندی سلسلهمراتبی.
        تهيئة التصنيف الهرمي.
        Initialiser le clustering hiérarchique.
        """
        self.data = data
        self.method = method
        self.threshold = threshold
        self.criterion = criterion
        self.linkage_matrix = None

    def fit_predict(self):
        """
        Apply Hierarchical clustering and return labels.
        اجرای خوشهبندی سلسلهمراتبی و بازگرداندن برچسبها.
        تطبيق التصنيف الهرمي وإرجاع التسميات.
        Appliquer le clustering hiérarchique et retourner les étiquettes.
        """
        self.linkage_matrix = linkage(self.data, method=self.method)
        return fcluster(self.linkage_matrix, t=self.threshold, criterion=self.criterion)


class MeanShiftClusterer:
    def init(self, data):
        """
        Initialize MeanShift clustering.
        مقداردهی اولیه خوشهبندی MeanShift.
        تهيئة MeanShift للتصنيف.
        Initialiser le clustering MeanShift.
        """
        self.data = data
        self.model = None

    def fit_predict(self):
        """
        Apply MeanShift clustering and return labels.
        اجرای خوشهبندی MeanShift و بازگرداندن برچسبها.
        تطبيق MeanShift وإرجاع التسميات.
        Appliquer MeanShift et retourner les étiquettes.
        """
        self.model = MeanShift()
        return self.model.fit_predict(self.data)


class AgglomerativeClusterer:
    def init(self, data, n_clusters=2):
        """
        Initialize Agglomerative clustering.
        مقداردهی اولیه خوشهبندی Agglomerative.
        تهيئة التصنيف التجميعي.
        Initialiser le clustering agglomératif.
        """
        self.data = data
        self.n_clusters = n_clusters
        self.model = None

    def fit_predict(self):
        """
        Apply Agglomerative clustering and return labels.
        اجرای خوشهبندی Agglomerative و بازگرداندن برچسبها.
        تطبيق التصنيف التجميعي وإرجاع التسميات.
        Appliquer le clustering agglomératif et retourner les étiquettes.
        """
        self.model = AgglomerativeClustering(n_clusters=self.n_clusters)
        return self.model.fit_predict(self.data)


def perform_clustering(data, method='kmeans', n_clusters=3, eps=0.5, min_samples=5, threshold=1.5):
    """
    Perform clustering on input data based on the selected method.
    اجرای خوشهبندی روی دادهها براساس روش انتخابشده.
    تنفيذ التصنيف بناءً على الطريقة المختارة.
    Effectuer le clustering en fonction de la méthode sélectionnée.
    """
    names = data['Name']
    features = data.drop(columns=['Name']).values

    if method == 'kmeans':
        clusterer = KMeansClusterer(features, n_clusters=n_clusters)
    elif method == 'dbscan':
        clusterer = DBSCANClusterer(features, eps=eps, min_samples=min_samples)
    elif method == 'hierarchical':
        clusterer = HierarchicalClusterer(features, method='ward', threshold=threshold)
    elif method == 'meanshift':
        clusterer = MeanShiftClusterer(features)
    elif method == 'agglomerative':
        clusterer = AgglomerativeClusterer(features, n_clusters=n_clusters)
    else:
        raise ValueError("Invalid clustering method. Choose 'kmeans', 'dbscan', 'hierarchical', 'meanshift', or 'agglomerative'.")

    labels = clusterer.fit_predict()
    output_data = pd.DataFrame({'Name': names, 'Cluster': labels})
    return output_data
    
def get_clustering_guide(language):
    """
    Returns a guide on when to use each clustering algorithm based on the specified language.
    بازگرداندن راهنما برای انتخاب روش خوشهبندی براساس زبان انتخابشده.
    إرجاع دليل لاختيار طريقة التصنيف بناءً على اللغة المحددة.
    Retourner un guide pour choisir la méthode de clustering en fonction de la langue spécifiée.
    """
    guides = {
        "english": {
            "kmeans": "KMeans is best suited for spherical clusters with similar sizes and requires the number of clusters as input.",
            "dbscan": "DBSCAN is ideal for detecting arbitrary-shaped clusters and noise, without needing the number of clusters.",
            "hierarchical": "Hierarchical clustering is suitable for hierarchical structures, with dendrograms to visualize relationships.",
            "meanshift": "MeanShift is good for finding clusters with a high-density region without needing the number of clusters.",
            "agglomerative": "Agglomerative clustering works well for building a hierarchy of clusters using a bottom-up approach."
},
        "farsi": {
            "kmeans": "KMeans برای خوشههای کروی با اندازههای مشابه و نیازمند تعداد خوشهها به عنوان ورودی مناسب است.",
            "dbscan": "DBSCAN برای شناسایی خوشههای با اشکال دلخواه و نویز، بدون نیاز به تعداد خوشهها ایدهآل است.",
            "hierarchical": "خوشهبندی سلسلهمراتبی برای ساختارهای سلسلهمراتبی مناسب است و دندروگرامها روابط را نمایش میدهند.",
            "meanshift": "MeanShift برای یافتن خوشههای با ناحیه تراکم بالا، بدون نیاز به تعداد خوشهها خوب است.",
            "agglomerative": "خوشهبندی Agglomerative برای ساخت سلسلهمراتب خوشهها از پایین به بالا مناسب است."
        },
        "arabic": {
            "kmeans": "KMeans مناسب لتصنيف المجموعات الكروية المتشابهة الحجم ويتطلب إدخال عدد المجموعات.",
            "dbscan": "DBSCAN مثالي لاكتشاف المجموعات ذات الأشكال التعسفية والضوضاء، دون الحاجة إلى عدد المجموعات.",
            "hierarchical": "التصنيف الهرمي مناسب للهياكل الهرمية، ويعرض العلاقات عبر المخططات الشجرية.",
            "meanshift": "MeanShift جيد لتحديد المجموعات ذات الكثافة العالية دون الحاجة إلى عدد المجموعات.",
            "agglomerative": "التصنيف التجميعي مناسب لبناء تسلسل هرمي من المجموعات باستخدام نهج من الأسفل إلى الأعلى."
        },
        "french": {
            "kmeans": "KMeans est idéal pour les clusters sphériques de taille similaire et nécessite le nombre de clusters en entrée.",
            "dbscan": "DBSCAN est idéal pour détecter des clusters de formes arbitraires et le bruit, sans nécessiter le nombre de clusters.",
            "hierarchical": "Le clustering hiérarchique est adapté aux structures hiérarchiques et utilise des dendrogrammes pour visualiser les relations.",
            "meanshift": "MeanShift est bien adapté pour trouver des clusters avec une région de forte densité sans nécessiter le nombre de clusters.",
            "agglomerative": "Le clustering agglomératif fonctionne bien pour construire une hiérarchie de clusters avec une approche ascendante."
        }
    }
    return guides.get(language.lower(), guides["english"])



@app.route('/cluster', methods=['POST'])
def cluster_data():
    """
    API endpoint to receive Excel file and perform clustering.
    نقطه پایانی API برای دریافت فایل اکسل و انجام خوشهبندی.
    نقطة النهاية API لاستلام ملف Excel وتنفيذ التصنيف.
    Point d'accès API pour recevoir le fichier Excel et effectuer le clustering.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    method = request.form.get('method', 'kmeans')
    n_clusters = int(request.form.get('n_clusters', 3))
    eps = float(request.form.get('eps', 0.5))
    min_samples = int(request.form.get('min_samples', 5))
    threshold = float(request.form.get('threshold', 1.5))

    data = pd.read_excel(file)

    try:
        result_df = perform_clustering(data, method=method, n_clusters=n_clusters, eps=eps, min_samples=min_samples, threshold=threshold)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    output_file = "clusters_output.xlsx"
    result_df.to_excel(output_file, index=False)

    return send_file(output_file, as_attachment=True)
    
    
@app.route('/clustering-guide', methods=['GET'])
def clustering_guide():
    language = request.args.get('language', 'english').lower()
    guide = get_clustering_guide(language)
    return jsonify(guide)
    

if __name__ == '__main__':
    app.run(debug=True)
