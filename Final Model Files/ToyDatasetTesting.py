# Noisy Data
csv_filename = "MXNWSS.csv"
df = pd.read_csv(csv_filename)

window_length = 5 
t_arr = df["value"].to_numpy().reshape(-1, 1)
SS = StandardScaler()

t_arr = SS.fit_transform(t_arr).flatten()

X = []
y = []

# print("Rearranging Data")
for i in range(len(t_arr)-window_length):
    t_row = []
    for j in t_arr[i:i+window_length]:
        t_row.append([j])
    X.append(t_row)
    y.append(t_arr[i + window_length])

X = np.array(X)
y = np.array(y)

# print(X.shape, y.shape)

X_train = X[:600]
X_valid = X[:800]
X_test = X[:1000]

y_train = y[:600]
y_valid = y[:800]
y_test = y[:1000]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.1)

# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, X_valid.shape, y_valid.shape)


model = Sequential([
    layers.InputLayer((window_length, 1)),
    layers.SimpleRNN(64, return_sequences=True),
    layers.SimpleRNN(64),
    layers.Dense(4),
    layers.Dense(1)
])

# print(model.summary())

cp = ModelCheckpoint("LSTMTestModel/", save_best_only=True)
model.compile(
    loss = MeanSquaredError(),
    optimizer = Adam(learning_rate=0.0001),
    metrics = [RootMeanSquaredError()]
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_valid, y_valid),
    epochs = 10,
    callbacks = [cp],
    verbose = 2
)


# LSTM
plt.plot(history.history["loss"], label="Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.plot(history.history["root_mean_squared_error"], label="RMSE")
plt.plot(history.history["val_root_mean_squared_error"], label="Val RMSE")
plt.title("Training History")
plt.legend()
plt.show()

view_length = 100
plt.figure(figsize=(10, 5))
plt.plot(SS.inverse_transform(model.predict(X_test))[:view_length], label="Model Output")
plt.plot(y_test[:view_length], label="Actual")
plt.title("Viewing Predictions")
plt.legend()
plt.show()


# RBM Playground
X, y = load_digits(return_X_y = True)

# X = np.asarray(X, "float32")
# X = minmax_scale(X, feature_range=(0, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.1, random_state = 0)

print(X_train.shape, X_valid.shape, X_test.shape, y_train.shape, y_valid.shape, y_test.shape)

# ("random forest", RandomForestClassifier()),
# ("ada boost", AdaBoostClassifier()),   
# ("gaussian process", GaussianProcessClassifier()),
# ("decision tree", DecisionTreeClassifier()),
# ("mlp", MLPClassifier()),
# ("svm", SVC())

# print("Before RBM: ", X_train[0])

pipe = Pipeline([
    ("mms1", MinMaxScaler()),
    ("rbm1", BernoulliRBM(n_components=640, learning_rate = 0.1, n_iter = 100, verbose = False, random_state = 0)),
    ("mms2", MinMaxScaler()),
    ("rbm2", BernoulliRBM(n_components=640, learning_rate = 0.1, n_iter = 100, verbose = False, random_state = 0)),
    ("mms3", MinMaxScaler()),
    ("rbm3", BernoulliRBM(n_components=640, learning_rate = 0.1, n_iter = 100, verbose = False, random_state = 0)),
    ("mms4", MinMaxScaler()),
    ("rbm4", BernoulliRBM(n_components=640, learning_rate = 0.1, n_iter = 100, verbose = False, random_state = 0)),
    ("mms5", MinMaxScaler()),
    ("rbm5", BernoulliRBM(n_components=640, learning_rate = 0.1, n_iter = 100, verbose = False, random_state = 0)),
    ("final_classifier", RandomForestClassifier(random_state = 0))
])
# print("After RBM: ",X_train[0])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print("Accuracy", accuracy_score(y_test, y_pred))

r_cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix = r_cm)

disp.plot()
plt.show()

# .91, .90, .91 -128, 10
# .89, .88, .90 -256, 10
# .93, .91, .90 -64, 10
# .89, .89, .92 -32, 10
# .85, .85, .85 -64, 64, (10)
# .86, .86, .86 -64, 64, (10)
# .88, .91, .87 -64, 64, (10, 20)
# .91, .91,

        # # Increasing
        # comb = []
        # for j in range(1, rbm_layer):
        #     comb.append((f"mms{j}", MinMaxScaler()))
        #     comb.append((f"rbm{j}", BernoulliRBM(n_components = X_train.shape[1] * j, learning_rate = 0.01, n_iter = 10, verbose = 0)))

        # comb.append((name, _clf))
        # predictor = Pipeline(comb)

        # predictor.fit(X_train, y_train)
        # y_pred = predictor.predict(X_test)
        # accuracy = accuracy_score(y_test, y_pred) * 100
        # Results[name]["INCREASING"][rbm_layer] = accuracy
        # print(f"{name}\tIncreasing\tLayer: {rbm_layer}\tAccuracy: {accuracy}")

        # # Decreasing
        # comb = []
        # for j in range(1, rbm_layer):
        #     comb.append((f"mms{j}", MinMaxScaler()))
        #     comb.append((f"rbm{j}", BernoulliRBM(n_components = X_train.shape[1] * rbm_layer - j, learning_rate = 0.01, n_iter = 10, verbose = 0)))

        # comb.append((name, _clf))
        # predictor = Pipeline(comb)

        # predictor.fit(X_train, y_train)
        # y_pred = predictor.predict(X_test)
        # accuracy = accuracy_score(y_test, y_pred) * 100
        # Results[name]["DECREASING"][rbm_layer] = accuracy
        # print(f"{name}\tDecreasing\tLayer: {rbm_layer}\tAccuracy: {accuracy}")

        
# Results = {}
# for name, _ in classifiers:
#     Results[name] = {
#         "CONSTANT" : {

#         },
#         "INCREASING" : {

#         },
#         "DECREASING" : {

#         }
#     }



# predictor.fit(X_train, y_train)
# y_pred = predictor.predict(X_test)

# pipe.fit(X_train, y_train)

# y_pred = pipe.predict(X_test)
# r_cm = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix = r_cm, display_labels = LE.classes_)

# with open("SciKitAccuracySaves.pkl", "wb") as f:
#     pickle.dump(Results, f)