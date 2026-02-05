# ============================================
# ANÁLISE ESTATÍSTICA DE DADOS DE E-COMMERCE
# ============================================

# IMPORTAÇÃO DAS BIBLIOTECAS
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Configuração visual
    sns.set(style="whitegrid")

    # ==============================
    # LEITURA DO ARQUIVO
    # ==============================
    df = pd.read_csv("ecommerce_estatistica.csv")

    # ==============================
    # SELEÇÃO DE COLUNAS
    # ==============================
    colunas_numericas = df.select_dtypes(include=['int64', 'float64']).columns
    colunas_categoricas = df.select_dtypes(include=['object']).columns

    num1 = colunas_numericas[0]
    num2 = colunas_numericas[1] if len(colunas_numericas) > 1 else colunas_numericas[0]
    cat1 = colunas_categoricas[0] if len(colunas_categoricas) > 0 else None

    # ==============================
    # 1️⃣ HISTOGRAMA
    # ==============================
    plt.figure(figsize=(8, 5))
    sns.histplot(df[num1], bins=30, kde=True)
    plt.title(f"Distribuição da variável {num1}")
    plt.xlabel(num1)
    plt.ylabel("Frequência")
    plt.show()

    # ==============================
    # 2️⃣ GRÁFICO DE DISPERSÃO
    # ==============================
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x=num1, y=num2, alpha=0.6)
    plt.title(f"Relação entre {num1} e {num2}")
    plt.xlabel(num1)
    plt.ylabel(num2)
    plt.show()

    # ==============================
    # 3️⃣ MAPA DE CALOR (CORRELAÇÃO)
    # ==============================
    plt.figure(figsize=(8, 6))
    correlacao = df[colunas_numericas].corr()
    sns.heatmap(correlacao, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Mapa de Calor das Correlações entre Variáveis Numéricas")
    plt.show()

    # ==============================
    # 4️⃣ GRÁFICO DE BARRAS
    # ==============================
    if cat1:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x=cat1)
        plt.title(f"Distribuição de Frequência da variável {cat1}")
        plt.xlabel(cat1)
        plt.ylabel("Quantidade")
        plt.xticks(rotation=45)
        plt.show()

    # ==============================
    # 5️⃣ GRÁFICO DE PIZZA
    # ==============================
    if cat1:
        plt.figure(figsize=(6, 6))
        df[cat1].value_counts().plot(
            kind='pie',
            autopct='%1.1f%%',
            startangle=90
        )
        plt.title(f"Proporção das Categorias da variável {cat1}")
        plt.ylabel("")
        plt.show()

    # ==============================
    # 6️⃣ GRÁFICO DE REGRESSÃO
    # ==============================
    plt.figure(figsize=(8, 5))
    sns.regplot(data=df, x=num1, y=num2)
    plt.title(f"Gráfico de Regressão entre {num1} e {num2}")
    plt.xlabel(num1)
    plt.ylabel(num2)
    plt.show()


# PONTO DE ENTRADA DO PROGRAMA
if __name__ == "__main__":
    main()
