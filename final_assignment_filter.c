#include "nn2.h"
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const int in1 = 784, out1 = 512;
const int in2 = out1, out2 = 256;
const int in3 = out2, out3 = 128;
const int in4 = out3, out4 = 10;

float *matrix1, *vector1, *matrix2, *vector2, *matrix3, *vector3, *matrix4, *vector4;

float learning_rate = 0.001f;
float beta1 = 0.9f;   // 一次の勾配(Momentom)の記憶率
float beta2 = 0.999f; // 二次の勾配(AdaGrad)の記憶率
const int mini_batch = 100;
const int epoc = 15;

// 複数のmallocを解放
void free_many(float **params, int count)
{
    for (int i = 0; i < count; i++)
    {
        free(params[i]);
        params[i] = NULL;
    }
}

// 正規分布に基づいた乱数関数の作成
double normrand()
{
    return sqrt(-2 * log((double)rand() / (RAND_MAX + 1.0))) *
           cos(2 * M_PI * ((double)rand() / (RAND_MAX + 1.0)));
}

// パラメータを乱数により初期化(He初期化を採用)
void reset_parameters(float *params, int insize, int outsize)
{
    float scale = sqrtf(2.0f / insize);
    for (int i = 0; i < insize * outsize; i++)
        params[i] = normrand() * scale;
}

// Fisher-Yatesのシャッフルアルゴリズムに基づいたtrain_xとtrain_y1のランダムな並べ替え
void shuffle_train_data(float *train_x, char *train_y, int train_count)
{
    for (int i = train_count - 1; i > 0; i--)
    {
        int j = rand() % (i + 1); // 0 〜 i のランダムな数

        // train_xのランダムな交換
        for (int k = 0; k < in1; k++)
        {
            float temp_x = train_x[i * in1 + k];
            train_x[i * in1 + k] = train_x[j * in1 + k];
            train_x[j * in1 + k] = temp_x;
        }

        // train_y1のtrain_xの交換に基づく交換
        char temp_y = train_y[i];
        train_y[i] = train_y[j];
        train_y[j] = temp_y;
    }
}

//  FC層の処理
void FC_filter(float *input, float *output, float *matrix, int insize, int outsize, float *vector, int mini_batch)
{
    for (int j = 0; j < mini_batch; j++)
    {
        for (int k = 0; k < outsize; k++)
            output[j * outsize + k] = vector[k]; // 初期化

        for (int n = 0; n < insize; n++)
        {
            for (int k = 0; k < outsize; k++)
            {
                output[j * outsize + k] += input[j * insize + n] * matrix[n + k * insize];
            }
        }
    }
}

//  ReLU層の処理
void ReLU_filter(float *input, float *output, int count)
{
    for (int i = 0; i < count; i++)
    {
        output[i] = (input[i] < 0) ? 0 : input[i];
    }
}

//  ソフトマックス関数の処理
void Sfmax(float *input, float *output, int outsize, int mini_batch)
{
    float *image_input;
    float max_val;

    for (int j = 0; j < mini_batch; j++)
    {
        image_input = &input[j * outsize];
        float *image_output = &output[j * outsize];
        max_val = image_input[0];

        for (int i = 1; i < outsize; i++)
        {
            if (image_input[i] > max_val)
                max_val = image_input[i];
        }

        float ex_sum = 0.0f;
        for (int i = 0; i < outsize; i++)
        {
            ex_sum += expf(image_input[i] - max_val); //  オーバーフローを防ぐために最大値を引いてからexpに載せる
        }

        for (int i = 0; i < outsize; i++)
        {
            image_output[i] = expf(image_input[i] - max_val) / ex_sum;
        }
    }
}

// 順伝播の処理
void forward_pass(float *input, float *output, float *relu1_input, float *fc2_input,
                  float *relu2_input, float *fc3_input, float *relu3_input, float *fc4_input, float *sfmax_input)
{
    FC_filter(input, relu1_input, matrix1, in1, out1, vector1, mini_batch);

    ReLU_filter(relu1_input, fc2_input, out1 * mini_batch);

    FC_filter(fc2_input, relu2_input, matrix2, in2, out2, vector2, mini_batch);

    ReLU_filter(relu2_input, fc3_input, out2 * mini_batch);

    FC_filter(fc3_input, relu3_input, matrix3, in3, out3, vector3, mini_batch);

    ReLU_filter(relu3_input, fc4_input, out3 * mini_batch);

    FC_filter(fc4_input, sfmax_input, matrix4, in4, out4, vector4, mini_batch);

    Sfmax(sfmax_input, output, out4, mini_batch);
}

// テスト用
void forward_pass_test(float *input, float *output, int test_count)
{
    float *input1 = malloc(sizeof(float) * out1 * test_count);
    float *input2 = malloc(sizeof(float) * out1 * test_count);
    float *input3 = malloc(sizeof(float) * out2 * test_count);
    float *input4 = malloc(sizeof(float) * out2 * test_count);
    float *input5 = malloc(sizeof(float) * out3 * test_count);
    float *input6 = malloc(sizeof(float) * out3 * test_count);
    float *input7 = malloc(sizeof(float) * out4 * test_count);

    FC_filter(input, input1, matrix1, in1, out1, vector1, test_count);

    ReLU_filter(input1, input2, out1 * test_count);

    FC_filter(input2, input3, matrix2, in2, out2, vector2, test_count);

    ReLU_filter(input3, input4, out2 * test_count);

    FC_filter(input4, input5, matrix3, in3, out3, vector3, test_count);

    ReLU_filter(input5, input6, out3 * test_count);

    FC_filter(input6, input7, matrix4, in4, out4, vector4, test_count);

    Sfmax(input7, output, out4, test_count);

    float *params[] = {input1, input2, input3, input4, input5, input6, input7};
    free_many(params, sizeof(params) / sizeof(params[0]));
}

// 正解ラベルをベクトル化
void set_answer(float *answer, char *answer_label, int outsize)
{
    for (int i = 0; i < mini_batch; i++)
    {
        for (int j = 0; j < outsize; j++)
        {
            answer[i * outsize + j] = answer_label[i] == j ? 1 : 0;
        }
    }
}

// ∂E(損失関数(sfMaxと統合))/∂xk(sfMaxへのインプット) = yk(sfMaxの出力) -tk(正解ベクトル)を導出
void backward_softmax_fc(float *grad_fc, float *sfmax_output, float *answer, int mini_batch, int outsize)
{
    for (int i = 0; i < outsize * mini_batch; i++)
    {
        grad_fc[i] = sfmax_output[i] - answer[i];
    }
}

// Relu層への勾配の計算
void backward_relu(float *grad_relu, float *grad_fc, float *fc_input, float *grad_mx, float *grad_v, int mini_batch,
                   float *matrix, int insize, int outsize)
{
    for (int i = 0; i < mini_batch; i++)
    {
        // 行列とベクトルの勾配を計算
        for (int j = 0; j < outsize; j++)
        {
            for (int n = 0; n < insize; n++)
            {
                grad_mx[j * insize + n] += (grad_fc[i * outsize + j] * fc_input[i * insize + n]) / mini_batch;
            }
            grad_v[j] += grad_fc[i * outsize + j] / mini_batch;
        }

        // Relu層への勾配の計算
        for (int n = 0; n < insize; n++)
        {
            grad_relu[i * insize + n] = 0;
            for (int j = 0; j < outsize; j++)
            {
                grad_relu[i * insize + n] += matrix[j * insize + n] * grad_fc[i * outsize + j];
            }
        }
    }
}

// FC層への勾配を計算
void backward_fc(float *grad_fc, float *grad_relu, float *relu_input, int mini_batch, int outsize)
{
    for (int i = 0; i < outsize * mini_batch; i++)
    {
        grad_fc[i] = relu_input[i] > 0 ? grad_relu[i] : 0;
    }
}

// 逆伝播の処理
void backward_pass(float *train_x, float *answer, float *final_output, float *sfmax_input, float *fc4_input,
                   float *relu3_input, float *fc3_input, float *relu2_input, float *fc2_input, float *relu1_input,
                   float *grad_fc4, float *grad_fc3, float *grad_fc2, float *grad_fc1, float *grad_relu3, float *grad_relu2, float *grad_relu1, float *grad_relu0, float *grad_mx4, float *grad_v4, float *grad_mx3, float *grad_v3, float *grad_mx2, float *grad_v2, float *grad_mx1, float *grad_v1)
{
    backward_softmax_fc(grad_fc4, final_output, answer, mini_batch, out4);

    backward_relu(grad_relu3, grad_fc4, fc4_input, grad_mx4, grad_v4, mini_batch, matrix4, in4, out4);

    backward_fc(grad_fc3, grad_relu3, relu3_input, mini_batch, out3);

    backward_relu(grad_relu2, grad_fc3, fc3_input, grad_mx3, grad_v3, mini_batch, matrix3, in3, out3);

    backward_fc(grad_fc2, grad_relu2, relu2_input, mini_batch, out2);

    backward_relu(grad_relu1, grad_fc2, fc2_input, grad_mx2, grad_v2, mini_batch, matrix2, in2, out2);

    backward_fc(grad_fc1, grad_relu1, relu1_input, mini_batch, out1);

    backward_relu(grad_relu0, grad_fc1, train_x, grad_mx1, grad_v1, mini_batch, matrix1, in1, out1);
}

// 行列やベクトルの勾配を初期化
void reset_grads(float *grad_mx, float *grad_v, int insize, int outsize)
{
    for (int i = 0; i < insize * outsize; i++)
        grad_mx[i] = 0;
    for (int i = 0; i < outsize; i++)
        grad_v[i] = 0;
}

// パラメータを更新(Adamモデルを使用)
void update_parameters(float *matrix, float *vector, float *grad_mx, float *grad_v, int insize, int outsize,
                       float *mom_mx, float *ada_mx, float *mom_v, float *ada_v, int t)
{
    float b1t = 1.0f - powf(beta1, t + 1);
    float b2t = 1.0f - powf(beta2, t + 1);

    for (int i = 0; i < insize * outsize; i++)
    {
        mom_mx[i] = beta1 * mom_mx[i] + (1 - beta1) * grad_mx[i];
        ada_mx[i] = beta2 * ada_mx[i] + (1 - beta2) * grad_mx[i] * grad_mx[i];

        float m_hat = mom_mx[i] / b1t;
        float v_hat = ada_mx[i] / b2t;

        matrix[i] -= learning_rate * m_hat / (sqrtf(v_hat) + 1e-7f);
    }

    for (int i = 0; i < outsize; i++)
    {
        mom_v[i] = beta1 * mom_v[i] + (1 - beta1) * grad_v[i];
        ada_v[i] = beta2 * ada_v[i] + (1 - beta2) * grad_v[i] * grad_v[i];

        float m_hat = mom_v[i] / b1t;
        float v_hat = ada_v[i] / b2t;

        vector[i] -= learning_rate * m_hat / (sqrtf(v_hat) + 1e-7f);
    }
}

void save_loss(float loss_average)
{
    FILE *fp = fopen("loss_history/loss_history.csv", "a"); // "a"は追記モード
    if (fp != NULL)
    {
        fprintf(fp, "%f\n", loss_average);
        fclose(fp);
    }

    printf("損失関数の平均 : %f\n", loss_average);
}

// 行列とベクトルのパラメータをファイルに保存
void save_parameters(const char *filename, int outsize, int insize, float *matrix, float *vector)
{
    FILE *fp = fopen(filename, "wb");
    assert(fp != NULL);

    fwrite(matrix, sizeof(float), insize * outsize, fp);
    fwrite(vector, sizeof(float), outsize, fp);

    fclose(fp);
}

// パラメータをファイルからロードする関数
void load_parameters(const char *filename, int out_size, int in_size, float *matrix, float *vector)
{
    FILE *fp = fopen(filename, "rb");
    assert(fp != NULL);

    // 行列 A の読み込み
    fread(matrix, sizeof(float), out_size * in_size, fp);

    // バイアス b の読み込み
    fread(vector, sizeof(float), out_size, fp);

    fclose(fp);
}

// 損失関数のミニバッチごとの平均を導出
float calc_loss(float *output, float *answer, int outsize, int mini_batch)
{
    float loss = 0.0f;
    for (int j = 0; j < mini_batch; j++)
    {
        for (int i = 0; i < outsize; i++)
        {
            loss -= answer[i + j * outsize] * logf(output[i + j * outsize] + 1e-7f);
        }
    }

    return loss / mini_batch;
}

// 出力からどの数字がどれくらいの確率で正解なのかを導出する関数
void correct_prediction(float *output, int *answer_num, float *max_prob)
{
    *answer_num = 0;
    *max_prob = output[0];
    for (int j = 1; j < out4; j++)
    {
        if (output[j] > *max_prob)
        {
            *max_prob = output[j];
            *answer_num = j;
        }
    }
}

// 結果を出力(解析した文字と正解の文字、テストの正答率)
float calc_accuracy(float *test_output, char *test_y, int test_count)
{
    float accuracy = 0.0f;

    for (int i = 0; i < test_count; i++)
    {
        float *output = &test_output[i * out4]; // 各テスト画像の出力部分だけ渡す
        int answer_num;
        float max_prob;

        correct_prediction(output, &answer_num, &max_prob);

        int label = (int)test_y[i];

        if (answer_num == label)
        {
            accuracy += 1.0f;
        }
    }

    return (accuracy / test_count) * 100;
}

// コンパイル後に実行する際の引数の数によって学習モードか判定モードかを分ける
int main(int argc, char *argv[])
{
    if (argc == 6)
    {
        float *input = load_mnist_bmp(argv[5]);
        float output[out4];

        matrix1 = malloc(sizeof(float) * in1 * out1);
        vector1 = calloc(out1, sizeof(float));
        matrix2 = malloc(sizeof(float) * in2 * out2);
        vector2 = calloc(out2, sizeof(float));
        matrix3 = malloc(sizeof(float) * in3 * out3);
        vector3 = calloc(out3, sizeof(float));
        matrix4 = malloc(sizeof(float) * in4 * out4);
        vector4 = calloc(out4, sizeof(float));

        load_parameters(argv[1], out1, in1, matrix1, vector1);
        load_parameters(argv[2], out2, in2, matrix2, vector2);
        load_parameters(argv[3], out3, in3, matrix3, vector3);
        load_parameters(argv[4], out4, in4, matrix4, vector4);

        forward_pass_test(input, output, 1);

        for (int i = 0; i < 10; i++)
        {
            printf("%d : %f\n", i, output[i] * 100);
        }

        int answer_num = 0;
        float max_prob = output[0];

        correct_prediction(output, &answer_num, &max_prob);

        printf("予測結果 : %d\n", answer_num);

        float *params[] = {input, matrix1, vector1, matrix2, vector2, matrix3, vector3, matrix4, vector4};
        free_many(params, sizeof(params) / sizeof(params[0]));
    }
    else
    {
        srand(time(NULL)); // 乱数初期化

        float *train_x1 = NULL;
        float *train_x2 = NULL;
        unsigned char *train_y1 = NULL;

        int train_count = -1;
        int train_count2 = -1;

        float *test_x = NULL;
        unsigned char *test_y = NULL;
        int test_count = -1;

        int width = -1;
        int height = -1;

        // nn2.hの関数を用いて訓練データとラベルなどを取得
        load_mnist(&train_x1, &train_x2, &train_y1, &train_count, &train_count2, &test_x, &test_y, &test_count, &width, &height);

        int total_augments = (train_count + train_count2) / train_count;
        float *train_x = malloc(sizeof(float) * width * height * train_count * total_augments);
        unsigned char *train_y = malloc(sizeof(unsigned char) * train_count * total_augments);

        int image_size = width * height;

        // train_xにまとめる
        for (int i = 0; i < train_count; i++)
        {
            memcpy(train_x + i * image_size, train_x1 + i * image_size, sizeof(float) * image_size);
            memcpy(train_x + (i + train_count) * image_size, train_x2 + i * image_size, sizeof(float) * image_size);
        }

        // train_yにまとめる
        for (int i = 0; i < train_count; i++)
        {
            for (int j = 0; j < total_augments; j++)
                train_y[i + train_count * j] = train_y1[i];
        }

        // 最後にtrain_countを調整する
        train_count *= total_augments;

        free(train_x1);
        free(train_x2);
        free(train_y1);

        // 行列とベクトルの配列をmallocでヒープ領域に確保
        matrix1 = malloc(sizeof(float) * in1 * out1);
        vector1 = calloc(out1, sizeof(float));
        matrix2 = malloc(sizeof(float) * in2 * out2);
        vector2 = calloc(out2, sizeof(float));
        matrix3 = malloc(sizeof(float) * in3 * out3);
        vector3 = calloc(out3, sizeof(float));
        matrix4 = malloc(sizeof(float) * in4 * out4);
        vector4 = calloc(out4, sizeof(float));

        reset_parameters(matrix1, in1, out1);
        reset_parameters(matrix2, in2, out2);
        reset_parameters(matrix3, in3, out3);
        reset_parameters(matrix4, in4, out4);

        // 各層の入力をmallocでヒープ領域に確保
        float *relu1_input = malloc(sizeof(float) * mini_batch * out1);
        float *fc2_input = malloc(sizeof(float) * mini_batch * out1);
        float *relu2_input = malloc(sizeof(float) * mini_batch * out2);
        float *fc3_input = malloc(sizeof(float) * mini_batch * out2);
        float *relu3_input = malloc(sizeof(float) * mini_batch * out3);
        float *fc4_input = malloc(sizeof(float) * mini_batch * out3);
        float *sfmax_input = malloc(sizeof(float) * mini_batch * out4);
        float *final_output = malloc(sizeof(float) * mini_batch * out4);

        // 正解ベクトル(t)を作成
        float *answer = malloc(sizeof(float) * mini_batch * out4);

        // 各関数の勾配はミニバッチ分確保し、行列とベクトルは平均を代入
        float *grad_fc4 = malloc(sizeof(float) * mini_batch * out4);
        float *grad_fc3 = malloc(sizeof(float) * mini_batch * out3);
        float *grad_fc2 = malloc(sizeof(float) * mini_batch * out2);
        float *grad_fc1 = malloc(sizeof(float) * mini_batch * out1);
        float *grad_relu3 = malloc(sizeof(float) * mini_batch * out3);
        float *grad_relu2 = malloc(sizeof(float) * mini_batch * out2);
        float *grad_relu1 = malloc(sizeof(float) * mini_batch * out1);
        float *grad_relu0 = malloc(sizeof(float) * mini_batch * in1);

        // 行列やベクトルの勾配をヒープ領域に確保
        float *grad_mx1 = malloc(in1 * out1 * sizeof(float));
        float *grad_v1 = malloc(out1 * sizeof(float));
        float *grad_mx2 = malloc(in2 * out2 * sizeof(float));
        float *grad_v2 = malloc(out2 * sizeof(float));
        float *grad_mx3 = malloc(in3 * out3 * sizeof(float));
        float *grad_v3 = malloc(out3 * sizeof(float));
        float *grad_mx4 = malloc(in4 * out4 * sizeof(float));
        float *grad_v4 = malloc(out4 * sizeof(float));

        // Momentumの勾配蓄積
        float *mom_mx1 = calloc(in1 * out1, sizeof(float));
        float *mom_v1 = calloc(out1, sizeof(float));
        float *mom_mx2 = calloc(in2 * out2, sizeof(float));
        float *mom_v2 = calloc(out2, sizeof(float));
        float *mom_mx3 = calloc(in3 * out3, sizeof(float));
        float *mom_v3 = calloc(out3, sizeof(float));
        float *mom_mx4 = calloc(in4 * out4, sizeof(float));
        float *mom_v4 = calloc(out4, sizeof(float));

        // アダグラッドの勾配蓄積
        float *ada_mx1 = calloc(in1 * out1, sizeof(float));
        float *ada_v1 = calloc(out1, sizeof(float));
        float *ada_mx2 = calloc(in2 * out2, sizeof(float));
        float *ada_v2 = calloc(out2, sizeof(float));
        float *ada_mx3 = calloc(in3 * out3, sizeof(float));
        float *ada_v3 = calloc(out3, sizeof(float));
        float *ada_mx4 = calloc(in4 * out4, sizeof(float));
        float *ada_v4 = calloc(out4, sizeof(float));

        float loss_average;

        float *test_output = malloc(sizeof(float) * out4 * test_count);

        for (int j = 0; j < epoc; j++)
        {
            loss_average = 0;

            shuffle_train_data(train_x, (char *)train_y, train_count);

            for (int i = 0; i < train_count / mini_batch; i++)
            {
                reset_grads(grad_mx1, grad_v1, in1, out1);
                reset_grads(grad_mx2, grad_v2, in2, out2);
                reset_grads(grad_mx3, grad_v3, in3, out3);
                reset_grads(grad_mx4, grad_v4, in4, out4);

                forward_pass(&train_x[i * in1 * mini_batch], final_output, relu1_input, fc2_input, relu2_input,
                             fc3_input, relu3_input, fc4_input, sfmax_input);

                set_answer(answer, (char *)&train_y[i * mini_batch], out4);

                backward_pass(&train_x[i * in1 * mini_batch], answer, final_output, sfmax_input, fc4_input, relu3_input, fc3_input, relu2_input, fc2_input, relu1_input, grad_fc4, grad_fc3, grad_fc2, grad_fc1, grad_relu3, grad_relu2, grad_relu1, grad_relu0, grad_mx4, grad_v4, grad_mx3, grad_v3, grad_mx2, grad_v2, grad_mx1, grad_v1);

                loss_average += calc_loss(final_output, answer, out4, mini_batch) / (train_count / mini_batch);

                update_parameters(matrix1, vector1, grad_mx1, grad_v1, in1, out1, mom_mx1, ada_mx1, mom_v1, ada_v1, i + j * (train_count / mini_batch));
                update_parameters(matrix2, vector2, grad_mx2, grad_v2, in2, out2, mom_mx2, ada_mx2, mom_v2, ada_v2, i + j * (train_count / mini_batch));
                update_parameters(matrix3, vector3, grad_mx3, grad_v3, in3, out3, mom_mx3, ada_mx3, mom_v3, ada_v3, i + j * (train_count / mini_batch));
                update_parameters(matrix4, vector4, grad_mx4, grad_v4, in4, out4, mom_mx4, ada_mx4, mom_v4, ada_v4, i + j * (train_count / mini_batch));
            }

            save_loss(loss_average);
        }

        float *params_1[] = {final_output, answer, grad_fc4, grad_fc3, grad_fc2, grad_fc1, grad_relu2, grad_relu1, grad_relu0, relu1_input, fc2_input, relu2_input, fc3_input, sfmax_input, grad_mx1, grad_v1, grad_mx2, grad_v2, grad_mx3, grad_v3, train_x, mom_mx1, mom_v1, mom_mx2, mom_v2, mom_mx3, mom_v3, ada_mx1, ada_v1, ada_mx2, ada_v2, ada_mx3, ada_v3};
        free_many(params_1, sizeof(params_1) / sizeof(params_1[0]));
        free(train_y);

        save_parameters("parameters/parameter_fc1.bin", out1, in1, matrix1, vector1);
        save_parameters("parameters/parameter_fc2.bin", out2, in2, matrix2, vector2);
        save_parameters("parameters/parameter_fc3.bin", out3, in3, matrix3, vector3);
        save_parameters("parameters/parameter_fc4.bin", out4, in4, matrix4, vector4);

        forward_pass_test(test_x, test_output, test_count);

        printf("正答率 : %.2f%%\n", calc_accuracy(test_output, (char *)test_y, test_count));

        float *params_2[] = {test_x, matrix1, vector1, matrix2, vector2, matrix3, vector3, test_output};
        free_many(params_2, sizeof(params_2) / sizeof(params_2[0]));
        free(test_y);

        float *params_3[] = {matrix4, vector4, grad_relu3, grad_mx4, grad_v4, mom_mx4, mom_v4, ada_mx4, ada_v4};
        free_many(params_3, sizeof(params_3) / sizeof(params_3[0]));
    }

/* 浮動小数点例外で停止することを確認するためのコード */
#if 0
    volatile float x = 0;
    volatile float y = 0;
    volatile float z = x/y;
#endif

    return 0;
}