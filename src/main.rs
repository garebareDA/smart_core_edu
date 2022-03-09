use smartcore::algorithm::neighbour::KNNAlgorithmName;
use smartcore::dataset::iris::load_dataset;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::neighbors::knn_classifier::KNNClassifier;
use smartcore::metrics::accuracy;

fn main() {
    let iris_data = load_dataset();
    let x = DenseMatrix::from_array(
        iris_data.num_samples,
        iris_data.num_features,
        &iris_data.data,
    );

    let y = iris_data.target;
    let knn = KNNClassifier::fit(
        &x,
        &y,
        Default::default(),
    ).unwrap();
    let y_hat = knn.predict(&x).unwrap();
    println!("accuracy: {}", accuracy(&y, &y_hat));
}
