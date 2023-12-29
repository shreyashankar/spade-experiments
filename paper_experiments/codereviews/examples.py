EXAMPLES = [
    {
        "pr_webhook_payload": '{"title": "Implement a basic Calculator class in Python", "description": "This PR introduces a Calculator class with basic arithmetic operations.", "author": "@pythonDev", "diff": [{"file": "calculator.py", "additions": "\\n+ class Calculator:\\n+     def __init__(self):\\n+         self.result = 0\\n+\\n+     def add(self, a, b):\\n+         self.result = a + b\\n+         return self.result\\n+\\n+     def subtract(self, a, b):\\n+         self.result = a - b\\n+         return self.result\\n+\\n+     def multiply(self, a, b):\\n+         self.result = a * b\\n+         return self.result\\n+\\n+     def divide(self, a, b):\\n+         if b != 0:\\n+             self.result = a / b\\n+             return self.result\\n+         else:\\n+             raise ValueError(\\"Cannot divide by zero\\")\\n+\\n+     def clear(self):\\n+         self.result = 0\\n+         return self.result\\n      ", "deletions": ""}]}'
    },
    {
        "pr_webhook_payload": '{"title": "Enhance user profile page in a JavaScript project", "description": "Adding new interactive elements to the user profile page.", "author": "@jsGuru", "diff": [{"file": "profile.js", "additions": "\\n+ document.addEventListener(\'DOMContentLoaded\', function() {\\n+     const profilePic = document.getElementById(\'profile-pic\');\\n+     profilePic.addEventListener(\'mouseover\', function() {\\n+         this.style.border = \'2px solid blue\';\\n+     });\\n+     profilePic.addEventListener(\'mouseout\', function() {\\n+         this.style.border = \'none\';\\n+     });\\n+\\n+     const bioText = document.getElementById(\'bio\');\\n+     bioText.addEventListener(\'click\', function() {\\n+         if (this.style.fontSize !== \'18px\') {\\n+             this.style.fontSize = \'18px\';\\n+         } else {\\n+             this.style.fontSize = \'14px\';\\n+         }\\n+     });\\n+\\n+     // More interactive elements here...\\n+ });\\n      ", "deletions": ""}]}'
    },
    {
        "pr_webhook_payload": '{"title": "Refactor login method in Java", "description": "Improving the login method for better error handling.", "author": "@javaExpert", "diff": [{"file": "Authenticator.java", "additions": "\\n+ public boolean login(String username, String password) {\\n+     if (username == null || username.isEmpty()) {\\n+         throw new IllegalArgumentException(\\"Username cannot be empty\\");\\n+     }\\n+     if (password == null || password.isEmpty()) {\\n+         throw new IllegalArgumentException(\\"Password cannot be empty\\");\\n+     }\\n+     User user = userRepository.findByUsername(username);\\n+     if (user == null) {\\n+         return false;\\n+     }\\n+     boolean passwordMatch = passwordService.verifyPassword(password, user.getPasswordHash());\\n+     if (!passwordMatch) {\\n+         return false;\\n+     }\\n+     sessionService.createSession(user);\\n+     return true;\\n+ }\\n      ", "deletions": "\\n- public boolean login(String username, String password) {\\n-     User user = userRepository.findByUsername(username);\\n-     if (user != null && user.getPassword().equals(password)) {\\n-         return true;\\n-     }\\n-     return false;\\n- }\\n      "}]}'
    },
    {
        "pr_webhook_payload": '{"title": "Add Vector3D class in C++", "description": "Implementing a 3D vector class with basic operations.", "author": "@cppEnthusiast", "diff": [{"file": "Vector3D.cpp", "additions": "\\n+ class Vector3D {\\n+ public:\\n+     Vector3D(float x, float y, float z) : x(x), y(y), z(z) {}\\n+     \\n+     float dot(const Vector3D& other) const {\\n+         return x * other.x + y * other.y + z * other.z;\\n+     }\\n+\\n+     Vector3D cross(const Vector3D& other) const {\\n+         return Vector3D(y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x);\\n+     }\\n+\\n+     Vector3D operator+(const Vector3D& other) const {\\n+         return Vector3D(x + other.x, y + other.y, z + other.z);\\n+     }\\n+\\n+     // Additional methods like subtract, multiply, etc...\\n+\\n+ private:\\n+     float x, y, z;\\n+ };\\n      ", "deletions": ""}]}'
    },
    {
        "pr_webhook_payload": '{"title": "Enhance User model with additional validations in Rails", "description": "Adding more validations to the User model in a Ruby on Rails application.", "author": "@rubyDev", "diff": [{"file": "user.rb", "additions": "\\n+ class User < ApplicationRecord\\n+   validates :name, presence: true, length: { minimum: 3 }\\n+   validates :email, presence: true, format: { with: URI::MailTo::EMAIL_REGEXP }\\n+   has_secure_password\\n+\\n+   def age\\n+     ((Time.zone.now - birthday.to_time) / 1.year.seconds).floor\\n+   end\\n+\\n+   def full_name\\n+     \\"#{first_name} #{last_name}\\"\\n+   end\\n+\\n+   # Additional methods and validations...\\n+ end\\n      ", "deletions": "\\n- class User < ApplicationRecord\\n-   validates :name, presence: true\\n-   validates :email, presence: true\\n-   has_secure_password\\n- end\\n      "}]}'
    },
    {
        "pr_webhook_payload": '{"title": "Add data cleaning functions in Python", "description": "Implementing data cleaning functions for a data analysis script.", "author": "@dataScientist", "diff": [{"file": "data_cleaning.py", "additions": "\\n+ import pandas as pd\\n+ import numpy as np\\n+\\n+ def clean_missing_values(df):\\n+     return df.replace({np.nan: None})\\n+\\n+ def standardize_column_names(df):\\n+     df.columns = [col.lower().replace(\' \', \'_\') for col in df.columns]\\n+     return df\\n+\\n+ def remove_outliers(df, column):\\n+     q1 = df[column].quantile(0.25)\\n+     q3 = df[column].quantile(0.75)\\n+     iqr = q3 - q1\\n+     lower_bound = q1 - 1.5 * iqr\\n+     upper_bound = q3 + 1.5 * iqr\\n+     return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]\\n+\\n+ # More data cleaning functions...\\n      ", "deletions": ""}]}'
    },
    {
        "pr_webhook_payload": '{"title": "Redesign homepage layout", "description": "Updating the HTML and CSS of the homepage for a modern look.", "author": "@webDesigner", "diff": [{"file": "index.html", "additions": "\\n+ <div class=\\"header\\">\\n+     <h1>Welcome to Our Site</h1>\\n+     <nav>\\n+         <ul>\\n+             <li><a href=\\"#\\">Home</a></li>\\n+             <li><a href=\\"#\\">About</a></li>\\n+             <li><a href=\\"#\\">Services</a></li>\\n+             <li><a href=\\"#\\">Contact</a></li>\\n+         </ul>\\n+     </nav>\\n+ </div>\\n+ <div class=\\"main\\">\\n+     <section>\\n+         <h2>Our Mission</h2>\\n+         <p>Lorem ipsum dolor sit amet...</p>\\n+     </section>\\n+     <!-- More sections here... -->\\n+ </div>\\n+ <div class=\\"footer\\">\\n+     <p>Copyright \\u00a9 2023</p>\\n+ </div>\\n      ", "deletions": "\\n- <div class=\\"container\\">\\n-     <h1>Old Site Header</h1>\\n-     <!-- Old HTML structure... -->\\n- </div>\\n      "}, {"file": "styles.css", "additions": "\\n+ .header {\\n+     background-color: #333;\\n+     color: white;\\n+     padding: 20px;\\n+ }\\n+ nav ul {\\n+     list-style-type: none;\\n+ }\\n+ nav ul li {\\n+     display: inline;\\n+     margin-right: 20px;\\n+ }\\n+ .main {\\n+     margin: 20px;\\n+ }\\n+ .footer {\\n+     background-color: #333;\\n+     color: white;\\n+     text-align: center;\\n+     padding: 10px;\\n+ }\\n      ", "deletions": "\\n- .container {\\n-     background-color: #f0f0f0;\\n-     color: black;\\n-     padding: 10px;\\n- }\\n- /* Old CSS styles... */\\n      "}]}'
    },
    {
        "pr_webhook_payload": '{"title": "Load dataset using pandas", "description": "Loading a CSV dataset into a pandas DataFrame.", "author": "@newDataSci", "diff": [{"file": "load_data.py", "additions": "\\n+ import pandas as pd\\n+ \\n+ data = pd.read_csv(\'data.csv\')\\n+ print(data.head())\\n      ", "deletions": ""}]}'
    },
    {
        "pr_webhook_payload": '{"title": "Add basic plot for data visualization", "description": "Creating a simple plot using matplotlib.", "author": "@plotBeginner", "diff": [{"file": "visualize_data.py", "additions": "\\n+ import matplotlib.pyplot as plt\\n+ import pandas as pd\\n+\\n+ data = pd.read_csv(\'data.csv\')\\n+ plt.plot(data[\'column1\'])\\n+ plt.show()\\n      ", "deletions": ""}]}'
    },
    {
        "pr_webhook_payload": '{"title": "Remove NaN values from DataFrame", "description": "Attempting to remove NaN values from a DataFrame.", "author": "@cleaningData", "diff": [{"file": "clean_data.py", "additions": "\\n+ import pandas as pd\\n+\\n+ data = pd.read_csv(\'data.csv\')\\n+ data = data.dropna()\\n+ print(data)\\n      ", "deletions": ""}]}'
    },
    {
        "pr_webhook_payload": '{"title": "Calculate mean and median of a dataset", "description": "Using pandas to calculate basic statistics of a dataset.", "author": "@statsNewbie", "diff": [{"file": "statistics.py", "additions": "\\n+ import pandas as pd\\n+\\n+ data = pd.read_csv(\'data.csv\')\\n+ print(\'Mean:\', data[\'column1\'].mean())\\n+ print(\'Median:\', data[\'column1\'].median())\\n      ", "deletions": ""}]}'
    },
    {
        "pr_webhook_payload": '{"title": "Train a basic linear regression model", "description": "Training a linear regression model using scikit-learn.", "author": "@mlStarter", "diff": [{"file": "train_model.py", "additions": "\\n+ from sklearn.model_selection import train_test_split\\n+ from sklearn.linear_model import LinearRegression\\n+ import pandas as pd\\n+\\n+ data = pd.read_csv(\'data.csv\')\\n+ X = data[[\'feature1\', \'feature2\']]\\n+ y = data[\'target\']\\n+ X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\\n+ model = LinearRegression()\\n+ model.fit(X_train, y_train)\\n+ print(\'Model trained successfully\')\\n      ", "deletions": ""}]}'
    },
    {
        "pr_webhook_payload": '{"title": "Add TCP connection establishment", "description": "Implementing basic TCP connection establishment using Rust\'s standard library.", "author": "@rustNetDev", "diff": [{"file": "tcp_connection.rs", "additions": "\\n+ use std::net::TcpStream;\\n+ use std::io::{self, Read, Write};\\n+\\n+ pub fn establish_connection(address: &str) -> io::Result<()> {\\n+     let mut stream = TcpStream::connect(address)?;\\n+     stream.write_all(b\\"Hello, server!\\")?;\\n+     let mut buffer = [0; 1024];\\n+     stream.read(&mut buffer)?;\\n+     println!(\\"Received response: {:?}\\", buffer);\\n+     Ok(())\\n+ }\\n      ", "deletions": ""}]}'
    },
    {
        "pr_webhook_payload": '{"title": "Create async server using Tokio", "description": "Building an asynchronous TCP server using the Tokio runtime.", "author": "@asyncRust", "diff": [{"file": "async_server.rs", "additions": "\\n+ use tokio::net::TcpListener;\\n+ use tokio::io::{AsyncReadExt, AsyncWriteExt};\\n+\\n+ #[tokio::main]\\n+ pub async fn run_server(address: &str) -> io::Result<()> {\\n+     let listener = TcpListener::bind(address).await?;\\n+     loop {\\n+         let (mut socket, _) = listener.accept().await?;\\n+         tokio::spawn(async move {\\n+             let mut buffer = [0; 1024];\\n+             socket.read(&mut buffer).await.unwrap();\\n+             socket.write_all(&buffer).await.unwrap();\\n+         });\\n+     }\\n+ }\\n      ", "deletions": ""}]}'
    },
    {
        "pr_webhook_payload": '{"title": "Integrate SSL/TLS support with rustls", "description": "Adding SSL/TLS support for secure communication using rustls crate.", "author": "@secureRust", "diff": [{"file": "tls_support.rs", "additions": "\\n+ use rustls::ServerConfig;\\n+ use std::sync::Arc;\\n+ use tokio_rustls::TlsAcceptor;\\n+\\n+ pub fn tls_config() -> Arc<ServerConfig> {\\n+     let config = ServerConfig::builder()\\n+         .with_safe_defaults()\\n+         .with_no_client_auth()\\n+         .with_cert_resolver(/* Resolver implementation */);\\n+     Arc::new(config)\\n+ }\\n+\\n+ // Function to establish a TLS connection...\\n      ", "deletions": ""}]}'
    },
    {
        "pr_webhook_payload": '{"title": "Implement basic TCP server", "description": "Adding a simple TCP server that listens on a specified port.", "author": "@rustNetworkDev", "diff": [{"file": "tcp_server.rs", "additions": "\\n+ use std::net::{TcpListener, TcpStream};\\n+ use std::io::{Read, Write};\\n+\\n+ fn handle_client(mut stream: TcpStream) {\\n+     let mut buffer = [0; 1024];\\n+     while match stream.read(&mut buffer) {\\n+         Ok(size) => {\\n+             stream.write(&buffer[0..size]).unwrap();\\n+             true\\n+         },\\n+         Err(_) => {\\n+             println!(\\"An error occurred, terminating connection with {}\\", stream.peer_addr().unwrap());\\n+             stream.shutdown(std::net::Shutdown::Both).unwrap();\\n+             false\\n+         }\\n+     } {}\\n+ }\\n+\\n+ fn main() {\\n+     let listener = TcpListener::bind(\\"127.0.0.1:7878\\").unwrap();\\n+     println!(\\"Server listening on port 7878\\");\\n+\\n+     for stream in listener.incoming() {\\n+         match stream {\\n+             Ok(stream) => {\\n+                 println!(\\"New connection: {}\\", stream.peer_addr().unwrap());\\n+                 std::thread::spawn(|| handle_client(stream));\\n+             }\\n+             Err(e) => {\\n+                 println!(\\"Error: {}\\", e);\\n+             }\\n+         }\\n+     }\\n+ }\\n      ", "deletions": ""}]}'
    },
    {
        "pr_webhook_payload": '{"title": "Implement WebSocket support", "description": "Adding support for WebSockets to handle real-time communication.", "author": "@webSocketPro", "diff": [{"file": "websocket.rs", "additions": "\\n+ use warp::Filter;\\n+\\n+ async fn handle_websocket(websocket: warp::ws::WebSocket) {\\n+     // WebSocket server logic here...\\n+ }\\n+\\n+ #[tokio::main]\\n+ async fn main() {\\n+     let websocket_route = warp::path(\\"ws\\")\\n+         .and(warp::ws())\\n+         .map(|ws: warp::ws::Ws| {\\n+             ws.on_upgrade(handle_websocket)\\n+         });\\n+\\n+     warp::serve(websocket_route)\\n+         .run(([127, 0, 0, 1], 3030))\\n+         .await;\\n+ }\\n      ", "deletions": ""}]}'
    },
    {
        "pr_webhook_payload": '{"title": "Setup Kubernetes client in Go", "description": "Adding functionality to initialize a Kubernetes client for interacting with the cluster.", "author": "@kubeDev", "diff": [{"file": "client.go", "additions": "\\n+ package main\\n+\\n+ import (\\n+     \\"k8s.io/client-go/kubernetes\\"\\n+     \\"k8s.io/client-go/rest\\"\\n+ )\\n+\\n+ func getKubernetesClient() (*kubernetes.Clientset, error) {\\n+     config, err := rest.InClusterConfig()\\n+     if err != nil {\\n+         return nil, err\\n+     }\\n+     clientset, err := kubernetes.NewForConfig(config)\\n+     if err != nil {\\n+         return nil, err\\n+     }\\n+     return clientset, nil\\n+ }\\n      ", "deletions": ""}]}'
    },
    {
        "pr_webhook_payload": '{"title": "Function to create a new secret", "description": "Implementing a function to create a new secret in Kubernetes using the Go client.", "author": "@secretManager", "diff": [{"file": "secrets.go", "additions": "\\n+ package main\\n+\\n+ import (\\n+     \\"context\\"\\n+     \\"k8s.io/client-go/kubernetes\\"\\n+     v1 \\"k8s.io/api/core/v1\\"\\n+     metav1 \\"k8s.io/apimachinery/pkg/apis/meta/v1\\"\\n+ )\\n+\\n+ func createSecret(clientset *kubernetes.Clientset, secretName string, data map[string]string) error {\\n+     secret := &v1.Secret{\\n+         ObjectMeta: metav1.ObjectMeta{\\n+             Name: secretName,\\n+         },\\n+         StringData: data,\\n+     }\\n+     _, err := clientset.CoreV1().Secrets(\\"default\\").Create(context.TODO(), secret, metav1.CreateOptions{})\\n+     return err\\n+ }\\n      ", "deletions": ""}]}'
    },
    {
        "pr_webhook_payload": '{"title": "Add functionality to read secret data", "description": "Creating a function to read data from a specified Kubernetes secret.", "author": "@goKube", "diff": [{"file": "read_secret.go", "additions": "\\n+ package main\\n+\\n+ import (\\n+     \\"context\\"\\n+     \\"k8s.io/client-go/kubernetes\\"\\n+     \\"fmt\\"\\n+ )\\n+\\n+ func readSecret(clientset *kubernetes.Clientset, secretName string) {\\n+     secret, err := clientset.CoreV1().Secrets(\\"default\\").Get(context.TODO(), secretName, metav1.GetOptions{})\\n+     if err != nil {\\n+         fmt.Println(err)\\n+         return\\n+     }\\n+     for key, value := range secret.Data {\\n+         fmt.Printf(\\"%s: %s\\n\\", key, value)\\n+     }\\n+ }\\n      ", "deletions": ""}]}'
    },
    {
        "pr_webhook_payload": '{"title": "Function to update an existing secret", "description": "Creating functionality to update an existing secret with new data.", "author": "@goDeveloper", "diff": [{"file": "update_secret.go", "additions": "\\n+ package main\\n+\\n+ import (\\n+     \\"context\\"\\n+     \\"k8s.io/client-go/kubernetes\\"\\n+     v1 \\"k8s.io/api/core/v1\\"\\n+     metav1 \\"k8s.io/apimachinery/pkg/apis/meta/v1\\"\\n+ )\\n+\\n+ func updateSecret(clientset *kubernetes.Clientset, secretName string, newData map[string]string) error {\\n+     secret, err := clientset.CoreV1().Secrets(\\"default\\").Get(context.TODO(), secretName, metav1.GetOptions{})\\n+     if err != nil {\\n+         return err\\n+     }\\n+     secret.StringData = newData\\n+     _, err = clientset.CoreV1().Secrets(\\"default\\").Update(context.TODO(), secret, metav1.UpdateOptions{})\\n+     return err\\n+ }\\n      ", "deletions": ""}]}'
    },
    {
        "pr_webhook_payload": '{"title": "Resolve race condition in Python threading", "description": "Fixing a race condition issue in a Python multithreaded application using threading.Lock.", "author": "@pythonConcurrent", "diff": [{"file": "multithreaded_app.py", "additions": "\\n+ import threading\\n+\\n+ class SharedResource:\\n+     def __init__(self):\\n+         self.resource = 0\\n+         self.lock = threading.Lock()\\n+\\n+     def update_resource(self, value):\\n+         with self.lock:\\n+             self.resource = value\\n      ", "deletions": "\\n- class SharedResource:\\n-     def __init__(self):\\n-         self.resource = 0\\n-\\n-     def update_resource(self, value):\\n-         self.resource = value\\n      "}]}'
    },
    {
        "pr_webhook_payload": '{"title": "Fix race condition in Java using synchronized method", "description": "Adding synchronization to methods to prevent race conditions in a Java application.", "author": "@javaThreadSafe", "diff": [{"file": "DataProcessor.java", "additions": "\\n+ public synchronized void processData(int data) {\\n+     // process data safely\\n+ }\\n      ", "deletions": "\\n- public void processData(int data) {\\n-     // process data\\n- }\\n      "}]}'
    },
    {
        "pr_webhook_payload": '{"title": "Resolve race condition in C++ using mutex", "description": "Implementing std::mutex to manage race conditions in a C++ concurrent application.", "author": "@cppMutex", "diff": [{"file": "resource_manager.cpp", "additions": "\\n+ #include <mutex>\\n+\\n+ std::mutex resource_mutex;\\n+\\n+ void updateResource() {\\n+     std::lock_guard<std::mutex> guard(resource_mutex);\\n+     // update shared resource safely\\n+ }\\n      ", "deletions": "\\n- void updateResource() {\\n-     // update shared resource\\n- }\\n      "}]}'
    },
    {
        "pr_webhook_payload": '{"title": "Use Go channels to fix race condition", "description": "Implementing Go channels to safely communicate between goroutines and prevent race conditions.", "author": "@goChannelExpert", "diff": [{"file": "data_collector.go", "additions": "\\n+ func collectData(dataChan chan<- Data) {\\n+     var data Data\\n+     // collect data\\n+     dataChan <- data // send data to channel\\n+ }\\n+\\n+ func main() {\\n+     dataChan := make(chan Data)\\n+     go collectData(dataChan)\\n+     // receive data from channel\\n+ }\\n      ", "deletions": "\\n- func collectData() Data {\\n-     var data Data\\n-     // collect data\\n-     return data\\n- }\\n-\\n- func main() {\\n-     data := collectData()\\n-     // use data\\n- }\\n      "}]}'
    },
    {
        "pr_webhook_payload": '{"title": "Fix race condition in Node.js using async/await", "description": "Utilizing async/await to handle asynchronous operations correctly and prevent race conditions.", "author": "@nodeAsyncFix", "diff": [{"file": "async_task.js", "additions": "\\n+ async function performTask() {\\n+     let result = await someAsyncOperation();\\n+     // process result\\n+ }\\n      ", "deletions": "\\n- function performTask() {\\n-     someAsyncOperation().then((result) => {\\n-         // process result\\n-     });\\n- }\\n      "}]}'
    },
    {
        "pr_webhook_payload": '{"title": "Optimize TensorFlow training loop", "description": "Improving the performance of the training loop in a TensorFlow model.", "author": "@tensorOptimizer", "diff": [{"file": "train_model.py", "additions": "\\n+ @tf.function\\n+ def train_step(model, inputs, labels, loss_function, optimizer):\\n+     with tf.GradientTape() as tape:\\n+         predictions = model(inputs, training=True)\\n+         loss = loss_function(labels, predictions)\\n+     gradients = tape.gradient(loss, model.trainable_variables)\\n+     optimizer.apply_gradients(zip(gradients, model.trainable_variables))\\n+     return loss\\n      ", "deletions": "\\n- def train_step(model, inputs, labels, loss_function, optimizer):\\n-     predictions = model(inputs, training=True)\\n-     loss = loss_function(labels, predictions)\\n-     gradients = model.optimizer.compute_gradients(loss)\\n-     model.optimizer.apply_gradients(gradients)\\n-     return loss\\n      "}]}'
    },
    {
        "pr_webhook_payload": '{"title": "Add early stopping to Keras model training", "description": "Introducing early stopping to Keras training to prevent overfitting.", "author": "@kerasDev", "diff": [{"file": "model_training.py", "additions": "\\n+ from keras.callbacks import EarlyStopping\\n+\\n+ early_stopping = EarlyStopping(monitor=\'val_loss\', patience=5)\\n+ model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, callbacks=[early_stopping])\\n      ", "deletions": "\\n- model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)\\n      "}]}'
    },
    {
        "pr_webhook_payload": '{"title": "Enhance PyTorch data loader efficiency", "description": "Optimizing the data loading pipeline in PyTorch for better training performance.", "author": "@pytorchEnhancer", "diff": [{"file": "data_loader.py", "additions": "\\n+ from torch.utils.data import DataLoader\\n+ from torchvision import transforms\\n+\\n+ transform = transforms.Compose([transforms.ToTensor(), ...])\\n+ train_dataset = CustomDataset(\'train_data/\', transform=transform)\\n+ train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=4)\\n      ", "deletions": "\\n- from torch.utils.data import DataLoader\\n- train_dataset = CustomDataset(\'train_data/\')\\n- train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)\\n      "}]}'
    },
    {
        "pr_webhook_payload": '{"title": "Create Flask endpoint for model serving", "description": "Setting up a Flask application to serve a trained machine learning model.", "author": "@modelServing", "diff": [{"file": "app.py", "additions": "\\n+ from flask import Flask, request, jsonify\\n+ import tensorflow as tf\\n+\\n+ app = Flask(__name__)\\n+ model = tf.keras.models.load_model(\'my_model.h5\')\\n+\\n+ @app.route(\'/predict\', methods=[\'POST\'])\\n+ def predict():\\n+     data = request.json\\n+     prediction = model.predict(data)\\n+     return jsonify(prediction.tolist())\\n+\\n+ if __name__ == \'__main__\':\\n+     app.run(debug=True)\\n      ", "deletions": ""}]}'
    },
    {
        "pr_webhook_payload": '{"title": "Implement Node.js server for TensorFlow.js model serving", "description": "Creating a server in Node.js to serve a TensorFlow.js model.", "author": "@nodeTfjs", "diff": [{"file": "server.js", "additions": "\\n+ const express = require(\'express\');\\n+ const tf = require(\'@tensorflow/tfjs-node\');\\n+ const app = express();\\n+ app.use(express.json());\\n+\\n+ let model;\\n+ tf.loadLayersModel(\'file://path/to/my-model.json\').then(loadedModel => {\\n+     model = loadedModel;\\n+ });\\n+\\n+ app.post(\'/predict\', async (req, res) => {\\n+     const inputData = tf.tensor2d(req.body.data);\\n+     const prediction = model.predict(inputData);\\n+     res.json(prediction.arraySync());\\n+ });\\n+\\n+ app.listen(3000, () => console.log(\'Server running on port 3000\'));\\n      ", "deletions": ""}]}'
    },
    {
        "pr_webhook_payload": '{"title": "Create basic ETL pipeline in Airflow", "description": "Setting up a simple ETL pipeline that extracts data from a source, transforms it, and loads it into a destination.", "author": "@dataPipeliner", "diff": [{"file": "etl_dag.py", "additions": "\\n+ from airflow import DAG\\n+ from airflow.operators.python_operator import PythonOperator\\n+ from datetime import datetime, timedelta\\n+\\n+ default_args = {\\n+     \'owner\': \'airflow\',\\n+     \'depends_on_past\': False,\\n+     \'start_date\': datetime(2023, 1, 1),\\n+     \'email_on_failure\': False,\\n+     \'email_on_retry\': False,\\n+     \'retries\': 1,\\n+     \'retry_delay\': timedelta(minutes=5),\\n+ }\\n+\\n+ dag = DAG(\'etl_pipeline\', default_args=default_args, schedule_interval=timedelta(days=1))\\n+\\n+ def extract():\\n+     # Code to extract data\\n+     pass\\n+\\n+ def transform():\\n+     # Code to transform data\\n+     pass\\n+\\n+ def load():\\n+     # Code to load data\\n+     pass\\n+\\n+ extract_task = PythonOperator(task_id=\'extract\', python_callable=extract, dag=dag)\\n+ transform_task = PythonOperator(task_id=\'transform\', python_callable=transform, dag=dag)\\n+ load_task = PythonOperator(task_id=\'load\', python_callable=load, dag=dag)\\n+\\n+ extract_task >> transform_task >> load_task\\n      ", "deletions": ""}]}'
    },
    {
        "pr_webhook_payload": '{"title": "Implement dynamic parameter handling in Airflow pipeline", "description": "Enhancing an Airflow pipeline to handle dynamic parameters for different runs.", "author": "@dynamicAirflow", "diff": [{"file": "dynamic_params_dag.py", "additions": "\\n+ from airflow.models import Variable\\n+\\n+ params = Variable.get(\\"my_dag_params\\", deserialize_json=True)\\n+ extract_task = PythonOperator(\\n+     task_id=\'extract\',\\n+     python_callable=extract,\\n+     op_kwargs={\'param\': params[\'extract_param\']},\\n+     dag=dag\\n+ )\\n      ", "deletions": "\\n- extract_task = PythonOperator(task_id=\'extract\', python_callable=extract, dag=dag)\\n      "}]}'
    },
    {
        "pr_webhook_payload": '{"title": "Add data quality checks to Airflow pipeline", "description": "Incorporating data quality checks into an existing Airflow pipeline.", "author": "@qualityCheck", "diff": [{"file": "data_quality_dag.py", "additions": "\\n+ from airflow.operators.python_operator import PythonOperator\\n+ from airflow.hooks.base_hook import BaseHook\\n+\\n+ def check_data_quality():\\n+     # Code to check data quality\\n+     pass\\n+\\n+ quality_check_task = PythonOperator(task_id=\'data_quality_check\', python_callable=check_data_quality, dag=dag)\\n+ load_task >> quality_check_task\\n      ", "deletions": ""}]}'
    },
    {
        "pr_webhook_payload": '{"title": "Integrate external data sources in Airflow pipeline", "description": "Modifying a pipeline to include data extraction from external sources.", "author": "@externalDataSource", "diff": [{"file": "external_source_dag.py", "additions": "\\n+ from airflow.providers.http.operators.http_operator import SimpleHttpOperator\\n+\\n+ extract_external_data_task = SimpleHttpOperator(\\n+     task_id=\'extract_external_data\',\\n+     http_conn_id=\'http_default\',\\n+     endpoint=\'data/api\',\\n+     method=\'GET\',\\n+     dag=dag\\n+ )\\n+ extract_task >> extract_external_data_task >> transform_task\\n      ", "deletions": "\\n- extract_task >> transform_task\\n      "}]}'
    },
    {
        "pr_webhook_payload": '{"title": "Create advanced analytics pipeline with Airflow", "description": "Setting up an Airflow pipeline for complex data processing and analytics tasks.", "author": "@analyticsGuru", "diff": [{"file": "analytics_dag.py", "additions": "\\n+ from airflow.operators.bash_operator import BashOperator\\n+ from airflow.operators.python_operator import PythonOperator\\n+\\n+ def perform_analysis():\\n+     # Complex data analysis code\\n+     pass\\n+\\n+ analysis_task = PythonOperator(task_id=\'perform_analysis\', python_callable=perform_analysis, dag=dag)\\n+ transform_task >> analysis_task\\n+ analysis_task >> load_task\\n      ", "deletions": "\\n- transform_task >> load_task\\n      "}]}'
    },
    {
        "pr_webhook_payload": '{"title": "Implement New Feature View Controller", "description": "Adding a new view controller to handle an advanced feature in the app.", "author": "@iosDev", "diff": [{"file": "FeatureViewController.swift", "additions": "\\n+ import UIKit\\n+\\n+ class FeatureViewController: UIViewController {\\n+     var featureModel: FeatureModel?\\n+\\n+     override func viewDidLoad() {\\n+         super.viewDidLoad()\\n+         // Additional setup after loading the view\\n+         configureView()\\n+     }\\n+\\n+     private func configureView() {\\n+         // View configuration code here\\n+     }\\n+\\n+     // Additional methods and logic for the new feature\\n+ }\\n+\\n+ // Extension for handling user interactions\\n+ extension FeatureViewController {\\n+     @IBAction func actionButtonPressed(_ sender: UIButton) {\\n+         // Handle button press\\n+     }\\n+ }\\n      ", "deletions": ""}]}'
    },
    {
        "pr_webhook_payload": '{"title": "Integrate Core Data for Local Persistence", "description": "Implementing Core Data to manage local data persistence in the app.", "author": "@coreDataPro", "diff": [{"file": "DataController.swift", "additions": "\\n+ import CoreData\\n+\\n+ class DataController {\\n+     static let shared = DataController()\\n+\\n+     let persistentContainer: NSPersistentContainer\\n+\\n+     private init() {\\n+         persistentContainer = NSPersistentContainer(name: \\"MyAppModel\\")\\n+         persistentContainer.loadPersistentStores { (storeDescription, error) in\\n+             if let error = error as NSError? {\\n+                 fatalError(\\"Unresolved error \\\\(error), \\\\(error.userInfo)\\")\\n+             }\\n+         }\\n+     }\\n+\\n+     // Core Data saving support\\n+     func saveContext () {\\n+         let context = persistentContainer.viewContext\\n+         if context.hasChanges {\\n+             do {\\n+                 try context.save()\\n+             } catch {\\n+                 let nserror = error as NSError\\n+                 fatalError(\\"Unresolved error \\\\(nserror), \\\\(nserror.userInfo)\\")\\n+             }\\n+         }\\n+     }\\n+ }\\n      ", "deletions": ""}]}'
    },
]
