{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Paipi11/Proyecto_control_calidad/blob/main/scripts/training/main.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import tensorflow as tf"
      ],
      "metadata": {
        "id": "8zzualbzcKGk"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "os.environ['CUDA_VISIBLE_DEVICES'] = '-1'"
      ],
      "metadata": {
        "id": "zLbH--JhIycX"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "data_dir = '/kaggle/working/fruit_ripeness_dataset/dataset/train'\n",
        "\n",
        "# Conjunto de entrenamiento (80%)\n",
        "train_dataset = tf.keras.utils.image_dataset_from_directory(\n",
        "    data_dir,\n",
        "    labels = 'inferred',\n",
        "    class_names = None,\n",
        "    validation_split=0.2,\n",
        "    subset=\"training\",\n",
        "    seed=42,\n",
        "    image_size=(128, 128),\n",
        "    batch_size=32,\n",
        "    color_mode='grayscale',\n",
        "    interpolation = 'bilinear',\n",
        "    follow_links = False,\n",
        "    crop_to_aspect_ratio = False,\n",
        "    label_mode='categorical',\n",
        "    shuffle=True\n",
        ")\n",
        "\n",
        "# Conjunto de validación (20%)\n",
        "val_dataset = tf.keras.utils.image_dataset_from_directory(\n",
        "    data_dir,\n",
        "    labels = 'inferred',\n",
        "    class_names = None,\n",
        "    validation_split=0.2,\n",
        "    subset=\"validation\",\n",
        "    seed=42,\n",
        "    image_size=(128, 128),\n",
        "    batch_size=32,\n",
        "    color_mode='grayscale',\n",
        "    interpolation = 'bilinear',\n",
        "    follow_links = False,\n",
        "    crop_to_aspect_ratio = False,\n",
        "    label_mode='categorical',\n",
        "    shuffle=True\n",
        ")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ccuJsHRxGb2d",
        "outputId": "74ca9990-1b1a-43ed-cb37-02dc44f0b09e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Found 14052 files belonging to 6 classes.\n",
            "Using 11242 files for training.\n",
            "Found 14052 files belonging to 6 classes.\n",
            "Using 2810 files for validation.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "for images, labels in train_dataset.take(1):\n",
        "    print(\"Image batch shape:\", images.shape)\n",
        "    print(\"Label batch shape:\", labels.shape)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "gktJYOV5JY_A",
        "outputId": "916994a5-3ba1-48a8-fe73-baba0be8a173"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Image batch shape: (32, 128, 128, 1)\n",
            "Label batch shape: (32, 6)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cnn_gray = tf.keras.models.Sequential()\n",
        "cnn_gray.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu',input_shape = [128,128,1])) # Se redimensiona la imagen y recibe la tupla [128,128,1] al tener solo una gama de colores 'grayscale'\n",
        "cnn_gray.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))\n",
        "cnn_gray.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu'))\n",
        "cnn_gray.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))\n",
        "cnn_gray.add(tf.keras.layers.Dropout(0.5))\n",
        "cnn_gray.add(tf.keras.layers.Flatten())\n",
        "cnn_gray.add(tf.keras.layers.Dense(units=128,activation='relu'))\n",
        "cnn_gray.add(tf.keras.layers.Dense(units = 6,activation='softmax')) # ultima capa con 6 unidades al ser 6 clases de clasificación."
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "wUX__hd5Gz0y",
        "outputId": "1c3d49aa-a3e3-4882-b1ec-22b760adde10"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.11/dist-packages/keras/src/layers/convolutional/base_conv.py:107: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.\n",
            "  super().__init__(activity_regularizer=activity_regularizer, **kwargs)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "epoch = 10"
      ],
      "metadata": {
        "id": "kDDtEGMDG5vt"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def reset_weights(model):\n",
        "    for layer in model.layers:\n",
        "        if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'bias_initializer'):\n",
        "            layer.kernel.assign(layer.kernel_initializer(tf.shape(layer.kernel)))\n",
        "            if layer.bias is not None:\n",
        "                layer.bias.assign(layer.bias_initializer(tf.shape(layer.bias)))"
      ],
      "metadata": {
        "id": "2sCqucyQG9uJ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "%%time\n",
        "reset_weights(cnn_gray)\n",
        "cnn_gray.compile(optimizer='SGD',loss='categorical_crossentropy',metrics=['accuracy'])\n",
        "training_history_SGD_grey = cnn_gray.fit(x=val_dataset, epochs=epoch)\n",
        "epochs = [i for i in range(1,epoch+1)]\n",
        "plt.plot(epochs,training_history_SGD_grey.history['accuracy'],color='red')\n",
        "plt.xlabel('No. Epochs')\n",
        "plt.ylabel('Training Accuracy')\n",
        "print(\"Precisiòn final del modelo: {} %\".format(training_history_SGD_grey.history['accuracy'][-1]*100))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 873
        },
        "id": "PXPTVAEEG6en",
        "outputId": "0d8e26d5-8c3b-4652-fb3e-06c3ff1b1d86"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/10\n",
            "\u001b[1m88/88\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m97s\u001b[0m 1s/step - accuracy: 0.1701 - loss: nan\n",
            "Epoch 2/10\n",
            "\u001b[1m88/88\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m87s\u001b[0m 983ms/step - accuracy: 0.1600 - loss: nan\n",
            "Epoch 3/10\n",
            "\u001b[1m88/88\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m87s\u001b[0m 988ms/step - accuracy: 0.1574 - loss: nan\n",
            "Epoch 4/10\n",
            "\u001b[1m88/88\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m87s\u001b[0m 982ms/step - accuracy: 0.1561 - loss: nan\n",
            "Epoch 5/10\n",
            "\u001b[1m88/88\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m144s\u001b[0m 1s/step - accuracy: 0.1513 - loss: nan\n",
            "Epoch 6/10\n",
            "\u001b[1m88/88\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m91s\u001b[0m 1s/step - accuracy: 0.1531 - loss: nan\n",
            "Epoch 7/10\n",
            "\u001b[1m88/88\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m137s\u001b[0m 972ms/step - accuracy: 0.1564 - loss: nan\n",
            "Epoch 8/10\n",
            "\u001b[1m88/88\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m143s\u001b[0m 989ms/step - accuracy: 0.1539 - loss: nan\n",
            "Epoch 9/10\n",
            "\u001b[1m88/88\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m140s\u001b[0m 968ms/step - accuracy: 0.1542 - loss: nan\n",
            "Epoch 10/10\n",
            "\u001b[1m88/88\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m87s\u001b[0m 984ms/step - accuracy: 0.1559 - loss: nan\n",
            "Precisiòn final del modelo: 15.978647768497467 %\n",
            "CPU times: user 20min 31s, sys: 3min 52s, total: 24min 23s\n",
            "Wall time: 18min 20s\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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wzA0atSo9qgRncmgQc0/v/qqeYmIMLceAABM4vRwXm5urp5++ml9+umnCgoK0ttvv61jx44pPT1dEydObI8a0ZmEhEj9+jWv799vbi0AAJjI6RC1d+9eTZs2TZLk5+enr7/+WqGhoXrxxRf1yiuvuL1AdEI8/gUAAOdDVEhIiH0eVExMjA4ePGjfdurUKfdVhs6LeVEAADg/J+qWW27RRx99pKFDh+qee+7RU089pU8//VTvvPOObrnllvaoEZ0NIQoAAOfPRL322mtKTU2VJL3wwgu68847tWbNGiUkJOjNN990uoDly5crISFBQUFBSk1NVVlZ2VX33bNnjyZMmKCEhARZLBYVFha2ul91dbWmTp2qiIgIBQcHa8SIEdq5c6d9+zvvvKO77rpLERERslgsqqiouKKPixcvavbs2YqIiFBoaKgmTJig2tpapz+fRyJEAQDgXIhqamrSF198obi4OEnNQ3srV67UP/7xD7399tuKj4936uBr1qxRTk6OFi1apF27dmnkyJEaP368Tp482er+Fy5c0IABA1RQUKDo6OhW9zlz5ozS0tLk7++vDRs2qLKyUkuWLFHPlme+Saqvr9ett956zTlc2dnZWrdunf785z/rgw8+0PHjx/XTn/7Uqc/nsVpC1P79Eg+kBgB4KYthGIYzbwgKCtLevXvVv3//6z54amqqxowZo2XLlkmSbDab+vXrp7lz52rBggXXfG9CQoKysrKUlZXl0L5gwQJt27ZNW7du/d7jHzlyRP3799ff//533XzzzfZ2q9WqPn366A9/+IMefPBBSdJnn32moUOHqqSkpM3DlnV1dQoPD5fValVYWFib3tMlXL4sdesmNTZKR4/+39V6AAB4gLZ+fzs9nDd8+HAdOnTouoqTpIaGBpWXlysjI+P/ivHxUUZGhkpKSlzud+3atUpOTtbEiRMVGRmppKQkrVq1yqk+ysvL1djY6FDbkCFDFBcXd83aLl26pLq6OofFI/n5SQMHNq9zhR4AwEs5HaJefvllPf3001q/fr1OnDjhcmg4deqUmpqaFBUV5dAeFRWlmpoaZ8uyO3TokFasWKHExERt3LhRs2bN0rx587R69eo291FTU6OAgAD16NHDqdry8/MVHh5uX/p58hka5kUBALxcm6/Oe/HFF/XUU0/pnnvukSTdd999slgs9u2GYchisaip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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data_dir = '/kaggle/working/fruit_ripeness_dataset/dataset/train'\n",
        "\n",
        "# Conjunto de entrenamiento (80%)\n",
        "train_dataset = tf.keras.utils.image_dataset_from_directory(\n",
        "    data_dir,\n",
        "    labels = 'inferred',\n",
        "    class_names = None,\n",
        "    validation_split=0.2,\n",
        "    subset=\"training\",\n",
        "    seed=42,\n",
        "    image_size=(128, 128),\n",
        "    batch_size=32,\n",
        "    color_mode='rgb',\n",
        "    interpolation = 'bilinear',\n",
        "    follow_links = False,\n",
        "    crop_to_aspect_ratio = False,\n",
        "    label_mode='categorical',\n",
        "    shuffle=True\n",
        ")\n",
        "\n",
        "# Conjunto de validación (20%)\n",
        "val_dataset = tf.keras.utils.image_dataset_from_directory(\n",
        "    data_dir,\n",
        "    labels = 'inferred',\n",
        "    class_names = None,\n",
        "    validation_split=0.2,\n",
        "    subset=\"validation\",\n",
        "    seed=42,\n",
        "    image_size=(128, 128),\n",
        "    batch_size=32,\n",
        "    color_mode='rgb',\n",
        "    interpolation = 'bilinear',\n",
        "    follow_links = False,\n",
        "    crop_to_aspect_ratio = False,\n",
        "    label_mode='categorical',\n",
        "    shuffle=True\n",
        ")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4rY5mBqePxZ5",
        "outputId": "a2a3c46c-63bf-4a4d-815e-9e3b0924cb0e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Found 14052 files belonging to 6 classes.\n",
            "Using 11242 files for training.\n",
            "Found 14052 files belonging to 6 classes.\n",
            "Using 2810 files for validation.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from keras import models, layers\n",
        "\n",
        "base_model = tf.keras.applications.MobileNetV2(\n",
        "    input_shape=(128, 128, 3),\n",
        "    include_top=False,\n",
        "    weights='imagenet'\n",
        ")\n",
        "base_model.trainable = False\n",
        "\n",
        "model = models.Sequential([\n",
        "    base_model,\n",
        "    layers.GlobalAveragePooling2D(),\n",
        "    layers.Dropout(0.5),\n",
        "    layers.Dense(128, activation='relu'),\n",
        "    layers.Dense(6, activation='softmax')\n",
        "])"
      ],
      "metadata": {
        "id": "E_vsHrAbfRjb"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model.compile(optimizer='adam',\n",
        "              loss='categorical_crossentropy',\n",
        "              metrics=['accuracy'])"
      ],
      "metadata": {
        "id": "BloTtF0vf47t"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "history = model.fit(\n",
        "    train_dataset,\n",
        "    validation_data=val_dataset,\n",
        "    epochs=10\n",
        ")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "V_SuMWhwgd6E",
        "outputId": "050d902e-0c9e-4044-fce2-bdbecee1a924"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/10\n",
            "\u001b[1m352/352\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m159s\u001b[0m 451ms/step - accuracy: 0.5956 - loss: 1.0399 - val_accuracy: 0.7548 - val_loss: 0.6747\n",
            "Epoch 2/10\n",
            "\u001b[1m352/352\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m184s\u001b[0m 401ms/step - accuracy: 0.6966 - loss: 0.7708 - val_accuracy: 0.7559 - val_loss: 0.6303\n",
            "Epoch 3/10\n",
            "\u001b[1m352/352\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m141s\u001b[0m 400ms/step - accuracy: 0.7157 - loss: 0.7243 - val_accuracy: 0.7705 - val_loss: 0.6019\n",
            "Epoch 4/10\n",
            "\u001b[1m352/352\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m153s\u001b[0m 432ms/step - accuracy: 0.7360 - loss: 0.6833 - val_accuracy: 0.7790 - val_loss: 0.5775\n",
            "Epoch 5/10\n",
            "\u001b[1m352/352\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m202s\u001b[0m 433ms/step - accuracy: 0.7484 - loss: 0.6589 - val_accuracy: 0.7705 - val_loss: 0.5871\n",
            "Epoch 6/10\n",
            "\u001b[1m352/352\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m201s\u001b[0m 431ms/step - accuracy: 0.7487 - loss: 0.6422 - val_accuracy: 0.7861 - val_loss: 0.5455\n",
            "Epoch 7/10\n",
            "\u001b[1m352/352\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m204s\u001b[0m 437ms/step - accuracy: 0.7621 - loss: 0.6251 - val_accuracy: 0.7747 - val_loss: 0.5596\n",
            "Epoch 8/10\n",
            "\u001b[1m352/352\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m143s\u001b[0m 407ms/step - accuracy: 0.7667 - loss: 0.6109 - val_accuracy: 0.7883 - val_loss: 0.5361\n",
            "Epoch 9/10\n",
            "\u001b[1m352/352\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m140s\u001b[0m 397ms/step - accuracy: 0.7644 - loss: 0.6012 - val_accuracy: 0.7947 - val_loss: 0.5327\n",
            "Epoch 10/10\n",
            "\u001b[1m352/352\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m156s\u001b[0m 437ms/step - accuracy: 0.7702 - loss: 0.5885 - val_accuracy: 0.7968 - val_loss: 0.5212\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "acc = history.history['accuracy']\n",
        "val_loss = history.history['val_loss']\n",
        "epochs = range(1, len(acc) + 1)\n",
        "\n",
        "fig, ax1 = plt.subplots(figsize=(10, 5))\n",
        "\n",
        "color = 'tab:blue'\n",
        "ax1.set_xlabel('Epoch')\n",
        "ax1.set_ylabel('Accuracy', color=color)\n",
        "ax1.plot(epochs, acc, color=color, label='Training Accuracy')\n",
        "ax1.tick_params(axis='y', labelcolor=color)\n",
        "\n",
        "ax2 = ax1.twinx()\n",
        "color = 'tab:red'\n",
        "ax2.set_ylabel('Validation Loss', color=color)\n",
        "ax2.plot(epochs, val_loss, color=color, label='Validation Loss')\n",
        "ax2.tick_params(axis='y', labelcolor=color)\n",
        "\n",
        "plt.title('Training Accuracy and Validation Loss vs Epochs')\n",
        "fig.tight_layout()\n",
        "plt.grid(True)\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 507
        },
        "id": "nAgNBDW8YA8o",
        "outputId": "01710878-ae7f-40d0-98f7-48c671d0f8ea"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1000x500 with 2 Axes>"
            ],
            "image/png": "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dK6IiIiIiIhIHVLhLSIiIiIiIlKHVHiLAHa7nUmTJmG3242OInJM+q5KY6LvqzQm+r5KY6HvauOke7xFRERERERE6pBmvEVERERERETqkApvERERERERkTqkwltERERERESkDqnwlibrySefpHv37vj5+REeHs4ll1zCb7/9ZnQskePy1FNPYTKZGDdunNFRRA6Tnp7O1VdfTUhICF5eXnTo0IHvv//e6Fgih3E6nUyYMIGEhAS8vLxo2bIljz32GNoCSRqCb7/9liFDhhAdHY3JZGLJkiU1Xne73UycOJGoqCi8vLwYOHAgO3bsMCas/C0V3tJkrVy5kltvvZV169axfPlyHA4H55xzDsXFxUZHEzmmjRs3MmvWLDp27Gh0FJHDHDhwgN69e2Oz2fj000/Ztm0bzz77LEFBQUZHEznM008/zYwZM3jxxRfZvn07Tz/9NM888wzTp083OpoIxcXFdOrUiZdeeumIrz/zzDO88MILzJw5k/Xr1+Pj48O5555LWVlZPSeV46FdzUUOysrKIjw8nJUrV9K3b1+j44gcUVFREV27duXll1/m8ccfp3Pnzjz33HNGxxKp9uCDD/Ldd9+xatUqo6OI/K0LL7yQiIgI5syZUz02dOhQvLy8ePPNNw1MJlKTyWRi8eLFXHLJJUDVbHd0dDT33HMP9957LwD5+flERETw+uuvM3LkSAPTypFoxlvkoPz8fACCg4MNTiJydLfeeisXXHABAwcONDqKyBEtW7aMbt26MXz4cMLDw+nSpQuzZ882OpbIEfXq1YuvvvqK33//HYCff/6Z1atXc9555xmcTOTYkpKS2Lt3b42fBwICAjj99NNZu3atgcnkaKxGBxBpCFwuF+PGjaN37960b9/e6DgiR/Tuu+/y448/snHjRqOjiBzV7t27mTFjBnfffTcPPfQQGzdu5I477sDDw4NRo0YZHU+khgcffJCCggLatGmDxWLB6XQydepUrrrqKqOjiRzT3r17AYiIiKgxHhERUf2aNCwqvEWomkXcsmULq1evNjqKyBGlpqZy5513snz5cjw9PY2OI3JULpeLbt268cQTTwDQpUsXtmzZwsyZM1V4S4OzYMEC3nrrLd5++23atWvHpk2bGDduHNHR0fq+ikit0lJzafJuu+02PvroI77++mtiY2ONjiNyRD/88AP79++na9euWK1WrFYrK1eu5IUXXsBqteJ0Oo2OKAJAVFQUiYmJNcbatm1LSkqKQYlEju6+++7jwQcfZOTIkXTo0IFrrrmGu+66iyeffNLoaCLHFBkZCcC+fftqjO/bt6/6NWlYVHhLk+V2u7nttttYvHgxK1asICEhwehIIkd19tln88svv7Bp06bqR7du3bjqqqvYtGkTFovF6IgiAPTu3fuw1oy///47zZo1MyiRyNGVlJRgNtf8cdhiseByuQxKJHJ8EhISiIyM5KuvvqoeKygoYP369fTs2dPAZHI0WmouTdatt97K22+/zdKlS/Hz86u+HyYgIAAvLy+D04nU5Ofnd9j+Az4+PoSEhGhfAmlQ7rrrLnr16sUTTzzBiBEj2LBhA6+88gqvvPKK0dFEDjNkyBCmTp1KfHw87dq146effmLatGlcf/31RkcToaioiJ07d1Y/T0pKYtOmTQQHBxMfH8+4ceN4/PHHOeWUU0hISGDChAlER0dX73wuDYvaiUmTZTKZjjj+2muvMXr06PoNI3ICzjrrLLUTkwbpo48+Yvz48ezYsYOEhATuvvtubrzxRqNjiRymsLCQCRMmsHjxYvbv3090dDRXXHEFEydOxMPDw+h40sR988039O/f/7DxUaNG8frrr+N2u5k0aRKvvPIKeXl59OnTh5dffplTTz3VgLTyd1R4i4iIiIiIiNQh3eMtIiIiIiIiUodUeIuIiIiIiIjUIRXeIiIiIiIiInVIhbeIiIiIiIhIHVLhLSIiIiIiIlKHVHiLiIiIiIiI1CEV3iIiIiIiIiJ1SIW3iIiIiIiISB1S4S0iItIEmEwmlixZYnQMERGRJkmFt4iISB0bPXo0JpPpsMfgwYONjiYiIiL1wGp0ABERkaZg8ODBvPbaazXG7Ha7QWlERESkPmnGW0REpB7Y7XYiIyNrPIKCgoCqZeAzZszgvPPOw8vLixYtWvD+++/XeP8vv/zCgAED8PLyIiQkhJtuuomioqIax8ydO5d27dpht9uJioritttuq/F6dnY2l156Kd7e3pxyyiksW7asbi9aREREABXeIiIiDcKECRMYOnQoP//8M1dddRUjR45k+/btABQXF3PuuecSFBTExo0bWbhwIV9++WWNwnrGjBnceuut3HTTTfzyyy8sW7aMVq1a1fiMKVOmMGLECDZv3sz555/PVVddRW5ubr1ep4iISFNkcrvdbqNDiIiInMxGjx7Nm2++iaenZ43xhx56iIceegiTycTYsWOZMWNG9WtnnHEGXbt25eWXX2b27Nk88MADpKam4uPjA8Ann3zCkCFDyMjIICIigpiYGK677joef/zxI2YwmUw88sgjPPbYY0BVMe/r68unn36qe81FRETqmO7xFhERqQf9+/evUVgDBAcHV/99z549a7zWs2dPNm3aBMD27dvp1KlTddEN0Lt3b1wuF7/99hsmk4mMjAzOPvvsY2bo2LFj9d/7+Pjg7+/P/v37T/SSRERE5Dip8BYREakHPj4+hy39ri1eXl7HdZzNZqvx3GQy4XK56iKSiIiI/Inu8RYREWkA1q1bd9jztm3bAtC2bVt+/vlniouLq1//7rvvMJvNtG7dGj8/P5o3b85XX31Vr5lFRETk+GjGW0REpB6Ul5ezd+/eGmNWq5XQ0FAAFi5cSLdu3ejTpw9vvfUWGzZsYM6cOQBcddVVTJo0iVGjRjF58mSysrK4/fbbueaaa4iIiABg8uTJjB07lvDwcM477zwKCwv57rvvuP322+v3QkVEROQwKrxFRETqwWeffUZUVFSNsdatW/Prr78CVTuOv/vuu9xyyy1ERUXxzjvvkJiYCIC3tzeff/45d955J927d8fb25uhQ4cybdq06nONGjWKsrIy/u///o97772X0NBQhg0bVn8XKCIiIkelXc1FREQMZjKZWLx4MZdcconRUURERKQO6B5vERERERERkTqkwltERERERESkDukebxEREYPpri8REZGTm2a8RUREREREROqQCm8RERERERGROqTCW0RERERERKQOqfAWERERERERqUMqvEVERERERETqkApvERERERERkTqkwltERERERESkDqnwFhEREREREalDKrxFRERERERE6tD/A8ULy0B6idRBAAAAAElFTkSuQmCC\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install keras-tuner --quiet"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "tv-8W8wLYTs3",
        "outputId": "50026870-8053-4b00-f82b-c992c3e2a97e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[?25l   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m0.0/129.1 kB\u001b[0m \u001b[31m?\u001b[0m eta \u001b[36m-:--:--\u001b[0m\r\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m129.1/129.1 kB\u001b[0m \u001b[31m5.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25h"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import keras_tuner as kt\n",
        "\n",
        "def build_model(hp):\n",
        "    base_model = tf.keras.applications.MobileNetV2(\n",
        "        input_shape=(128, 128, 3),\n",
        "        include_top=False,\n",
        "        weights='imagenet'\n",
        "    )\n",
        "    base_model.trainable = False\n",
        "\n",
        "    model = tf.keras.Sequential()\n",
        "    model.add(base_model)\n",
        "    model.add(layers.GlobalAveragePooling2D())\n",
        "\n",
        "    # Número de neuronas\n",
        "    hp_units = hp.Int('units', min_value=64, max_value=256, step=64)\n",
        "    model.add(layers.Dense(units=hp_units, activation='relu'))\n",
        "\n",
        "    # Tasa de dropout\n",
        "    hp_dropout = hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)\n",
        "    model.add(layers.Dropout(hp_dropout))\n",
        "\n",
        "    model.add(layers.Dense(6, activation='softmax'))\n",
        "\n",
        "    # Tasa de aprendizaje\n",
        "    hp_learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])\n",
        "\n",
        "    model.compile(\n",
        "        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),\n",
        "        loss='categorical_crossentropy',\n",
        "        metrics=['accuracy']\n",
        "    )\n",
        "    return model"
      ],
      "metadata": {
        "id": "Vz2vNLqMYYkx"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "tuner = kt.RandomSearch(\n",
        "    build_model,\n",
        "    objective='val_accuracy',\n",
        "    max_trials=10,\n",
        "    directory='tuning_dir',\n",
        "    project_name='fruit_tuning'\n",
        ")"
      ],
      "metadata": {
        "id": "q723v3_CYeOq"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "tuner.search(train_dataset, validation_data=val_dataset, epochs=5)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "5FMU3GHlYkuS",
        "outputId": "510dfaaa-3734-419d-afe1-44abb54db223"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Trial 10 Complete [00h 16m 12s]\n",
            "val_accuracy: 0.7829181551933289\n",
            "\n",
            "Best val_accuracy So Far: 0.8234875202178955\n",
            "Total elapsed time: 02h 24m 22s\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "best_model = tuner.get_best_models(num_models=1)[0]\n",
        "best_hyperparams = tuner.get_best_hyperparameters(1)[0]\n",
        "print(\"Mejores hiperparámetros:\")\n",
        "print(best_hyperparams.values)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "u8jeMlYYYrHj",
        "outputId": "8488270a-adcc-462c-9848-341b099c5faa"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mejores hiperparámetros:\n",
            "{'units': 192, 'dropout': 0.30000000000000004, 'learning_rate': 0.001}\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_lib.py:757: UserWarning: Skipping variable loading for optimizer 'adam', because it has 2 variables whereas the saved optimizer has 10 variables. \n",
            "  saveable.load_own_variables(weights_store.get(inner_path))\n"
          ]
        }
      ]
    }
  ]
}