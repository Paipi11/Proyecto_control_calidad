{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPbj2y3XbWI1DADoUAMlU3F",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Paipi11/Proyecto_control_calidad/blob/main/scripts/data_acquisition/main.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **Carga o Adquisición de Datos**\n",
        "---\n",
        "\n",
        "Para la carga de datos, se utilizó la API de Kaggle para obtener los datos directamente."
      ],
      "metadata": {
        "id": "Y2ZEygzYZK3j"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "E6udLSEZXPOI",
        "outputId": "9f4fd961-1fc2-4141-bd77-51c5b867dd95"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: kagglehub in /usr/local/lib/python3.11/dist-packages (0.3.12)\n",
            "Requirement already satisfied: packaging in /usr/local/lib/python3.11/dist-packages (from kagglehub) (24.2)\n",
            "Requirement already satisfied: pyyaml in /usr/local/lib/python3.11/dist-packages (from kagglehub) (6.0.2)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.11/dist-packages (from kagglehub) (2.32.3)\n",
            "Requirement already satisfied: tqdm in /usr/local/lib/python3.11/dist-packages (from kagglehub) (4.67.1)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests->kagglehub) (3.4.2)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests->kagglehub) (3.10)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests->kagglehub) (2.4.0)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests->kagglehub) (2025.6.15)\n"
          ]
        }
      ],
      "source": [
        "!pip install kagglehub"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Cargamos las librerías necesarias para realizar los filtros y depuración de los datos"
      ],
      "metadata": {
        "id": "dkz0BaQJZShH"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import os\n",
        "import kagglehub\n",
        "import shutil\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "from pathlib import Path\n",
        "from PIL import Image"
      ],
      "metadata": {
        "id": "8KAMmlLpXluy"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ruta = kagglehub.dataset_download(\"leftin/fruit-ripeness-unripe-ripe-and-rotten\")\n",
        "ruta_destino = \"/kaggle/working/fruit_dataset_editable\"\n",
        "\n",
        "if os.path.exists(ruta_destino):\n",
        "    shutil.rmtree(ruta_destino)\n",
        "\n",
        "shutil.copytree(ruta, ruta_destino)\n",
        "print(f\"Dataset copiado a: {ruta_destino}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "VH6K9z5CXssD",
        "outputId": "17807e43-bf9f-440d-fe0b-ff1c350690df"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Downloading from https://www.kaggle.com/api/v1/datasets/download/leftin/fruit-ripeness-unripe-ripe-and-rotten?dataset_version_number=1...\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|██████████| 3.64G/3.64G [00:41<00:00, 93.5MB/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Extracting files...\n",
            "Dataset copiado a: /kaggle/working/fruit_dataset_editable\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Se quiere observar la jerarquía de las carpetas\n",
        "\n",
        "**Nota:**\n",
        "Tener presente que solo utilizaremos los folders que tienen el identificador de **fresh** y **rotten**. El de **unripe** se descartará del modelo."
      ],
      "metadata": {
        "id": "Q--DbNepZmw1"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def mostr(carpeta, n=0):\n",
        "    esp = \"  \" * n +\"-\"\n",
        "    for i in sorted(os.listdir(carpeta)):\n",
        "        ruta = os.path.join(carpeta, i)\n",
        "        if os.path.isdir(ruta):\n",
        "            print(esp + i)\n",
        "            mostr(ruta, n +1)\n",
        "mostr(ruta_destino)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Hbe3TJ56X0Ax",
        "outputId": "2d96ed63-f598-44a3-ff4b-7c9790f9b180"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "-archive (1)\n",
            "  -dataset\n",
            "    -dataset\n",
            "      -test\n",
            "        -freshapples\n",
            "        -freshbanana\n",
            "        -freshoranges\n",
            "        -rottenapples\n",
            "        -rottenbanana\n",
            "        -rottenoranges\n",
            "        -unripe apple\n",
            "        -unripe banana\n",
            "        -unripe orange\n",
            "      -train\n",
            "        -freshapples\n",
            "        -freshbanana\n",
            "        -freshoranges\n",
            "        -rottenapples\n",
            "        -rottenbanana\n",
            "        -rottenoranges\n",
            "        -unripe apple\n",
            "        -unripe banana\n",
            "        -unripe orange\n",
            "    -test\n",
            "      -freshapples\n",
            "      -freshbanana\n",
            "      -freshoranges\n",
            "      -rottenapples\n",
            "      -rottenbanana\n",
            "      -rottenoranges\n",
            "      -unripe apple\n",
            "      -unripe banana\n",
            "      -unripe orange\n",
            "    -train\n",
            "      -freshapples\n",
            "      -freshbanana\n",
            "      -freshoranges\n",
            "      -rottenapples\n",
            "      -rottenbanana\n",
            "      -rottenoranges\n",
            "      -unripe apple\n",
            "      -unripe banana\n",
            "      -unripe orange\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Se reorganiza la jerarquía de las carpetas introduciendo el folder \"dataset 2\""
      ],
      "metadata": {
        "id": "Ie2cLTanZ3hU"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "base_ruth = os.path.join(ruta_destino, 'archive (1)', 'dataset')\n",
        "carpeta_test = os.path.join(base_ruth, 'test')\n",
        "carpeta_train = os.path.join(base_ruth, 'train')\n",
        "conjunto_01 = os.path.join(base_ruth, 'dataset 2')\n",
        "shutil.move(carpeta_test, os.path.join(conjunto_01, 'test'))\n",
        "shutil.move(carpeta_train, os.path.join(conjunto_01, 'train'))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 36
        },
        "id": "Gh8jWEOxYkGm",
        "outputId": "fdaf865a-a713-4606-b9e0-661277f66480"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "'/kaggle/working/fruit_dataset_editable/archive (1)/dataset/dataset 2/train'"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            }
          },
          "metadata": {},
          "execution_count": 5
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Al conocer que el folder \"dataset 2\" es igual a \"dataset 1\" se procede a eliminarla y a su vez se eliminan las carpetas de **unripe** que están fuera del objetivo del proyecto."
      ],
      "metadata": {
        "id": "wSkGveRDaH-0"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import shutil\n",
        "from pathlib import Path\n",
        "\n",
        "shutil.rmtree(\"/kaggle/working/fruit_dataset_editable/archive (1)/dataset/dataset 2\")\n",
        "\n",
        "directorio_01 = Path('/kaggle/working/fruit_dataset_editable/archive (1)/dataset/dataset/test/unripe apple')\n",
        "directorio_02 = Path('/kaggle/working/fruit_dataset_editable/archive (1)/dataset/dataset/test/unripe banana')\n",
        "directorio_03 = Path('/kaggle/working/fruit_dataset_editable/archive (1)/dataset/dataset/test/unripe orange')\n",
        "\n",
        "shutil.rmtree(directorio_01)\n",
        "shutil.rmtree(directorio_02)\n",
        "shutil.rmtree(directorio_03)\n",
        "\n",
        "directorio_01_1 = Path('/kaggle/working/fruit_dataset_editable/archive (1)/dataset/dataset/train/unripe apple')\n",
        "directorio_02_2 = Path('/kaggle/working/fruit_dataset_editable/archive (1)/dataset/dataset/train/unripe banana')\n",
        "directorio_03_3 = Path('/kaggle/working/fruit_dataset_editable/archive (1)/dataset/dataset/train/unripe orange')\n",
        "\n",
        "shutil.rmtree(directorio_01_1)\n",
        "shutil.rmtree(directorio_02_2)\n",
        "shutil.rmtree(directorio_03_3)"
      ],
      "metadata": {
        "id": "WjIsT5HbYn_d"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Finalmente, se eliminan las 27 imagenes que no cumplen con el criterio de 128x128 pixeles como mínimo."
      ],
      "metadata": {
        "id": "-2i7wR9OamX4"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "ancho_min = 127\n",
        "alto_min = 127\n",
        "\n",
        "t = 0\n",
        "for r, d, f in os.walk(base_ruth):\n",
        "    for file in f:\n",
        "        ruta_imagen = os.path.join(r, file)\n",
        "\n",
        "        with Image.open(ruta_imagen) as img:\n",
        "                ancho, alto = img.size\n",
        "                if ancho < ancho_min or alto < alto_min:\n",
        "                    img.close()\n",
        "                    os.remove(ruta_imagen)\n",
        "                    t += 1\n",
        "t"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ea2hCzorYrOq",
        "outputId": "d9f3ac28-a7fb-4e32-b058-9fdd26f9edfc"
      },
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "27"
            ]
          },
          "metadata": {},
          "execution_count": 9
        }
      ]
    }
  ]
}