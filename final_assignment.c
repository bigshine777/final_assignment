#include "nn.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const int in1 = 784, out1 = 128;
const int in2 = out1, out2 = 256;
const int in3 = out2, out3 = 10;

const float learning_rate = 1.0f;
const int mini_batch = 100;
const int epoc = 20;

// 複数のmallocを解放
void free_many(float **ptrs, int count)
{
    for (int i = 0; i < count; i++)
    {
        free(ptrs[i]);
    }
}

// 正規分布に基づいた乱数関数の作成
double normrand()
{
    return sqrt(-2 * log((double)rand() / (RAND_MAX + 1.0))) *
           cos(2 * M_PI * ((double)rand() / (RAND_MAX + 1.0)));
}

// Fisher-Yatesのシャッフルアルゴリズムに基づいたtrain_xとtrain_yのランダムな並べ替え
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

        // train_yのランダムな交換
        char tmp_y = train_y[i];
        train_y[i] = train_y[j];
        train_y[j] = tmp_y;
    }
}

// パラメータを乱数により初期化
void reset_parameters(float *matrix1, float *vector1,
                      float *matrix2, float *vector2,
                      float *matrix3, float *vector3)
{

    for (int i = 0; i < in1 * out1; i++)
        matrix1[i] = normrand() * 0.01f;

    for (int i = 0; i < out1; i++)
        vector1[i] = 0.0f;

    for (int i = 0; i < in2 * out2; i++)
        matrix2[i] = normrand() * 0.01f;

    for (int i = 0; i < out2; i++)
        vector2[i] = 0.0f;

    for (int i = 0; i < in3 * out3; i++)
        matrix3[i] = normrand() * 0.01f;

    for (int i = 0; i < out3; i++)
        vector3[i] = 0.0f;
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
void ReLU_filter(float *input, float *output, int outsize)
{
    for (int i = 0; i < outsize; i++)
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
void forward_pass(float *input, float *output, float *matrix1, float *vector1, float *matrix2, float *vector2, float *matrix3, float *vector3, float *relu1_input, float *fc2_input, float *relu2_input, float *fc3_input, float *sfmax_input, int mini_batch)
{
    FC_filter(input, relu1_input, matrix1, in1, out1, vector1, mini_batch);

    ReLU_filter(relu1_input, fc2_input, out1 * mini_batch);

    FC_filter(fc2_input, relu2_input, matrix2, in2, out2, vector2, mini_batch);

    ReLU_filter(relu2_input, fc3_input, out2 * mini_batch);

    FC_filter(fc3_input, sfmax_input, matrix3, in3, out3, vector3, mini_batch);

    Sfmax(sfmax_input, output, out3, mini_batch);
}

// テスト用
void forward_pass_test(float *input, float *output, float *matrix1, float *vector1, float *matrix2, float *vector2, float *matrix3, float *vector3, int test_count)
{
    float *input1 = malloc(sizeof(float) * out1 * test_count);
    float *input2 = malloc(sizeof(float) * out1 * test_count);
    float *input3 = malloc(sizeof(float) * out2 * test_count);
    float *input4 = malloc(sizeof(float) * out2 * test_count);
    float *input5 = malloc(sizeof(float) * out3 * test_count);

    FC_filter(input, input1, matrix1, in1, out1, vector1, test_count);

    ReLU_filter(input1, input2, out1 * test_count);

    FC_filter(input2, input3, matrix2, in2, out2, vector2, test_count);

    ReLU_filter(input3, input4, out2 * test_count);

    FC_filter(input4, input5, matrix3, in3, out3, vector3, test_count);

    Sfmax(input5, output, out3, test_count);

    float *params_1[] = {input1, input2, input3, input4, input5};
    free_many(params_1, sizeof(params_1) / sizeof(params_1[0]));
}

// 正解ラベルをベクトル化
void set_answer(float *answer, char *answer_label, int mini_batch, int outsize)
{
    for (int i = 0; i < mini_batch; i++)
    {
        for (int j = 0; j < outsize; j++)
        {
            answer[i * outsize + j] = answer_label[i] == j ? 1 : 0;
        }
    }
}

// ∂E(損失関数(sfMaxと統合))/∂xk(sfMaxへのインプット) = yk(sfMaxの出力) -tk(正解ベクトル)
void backward_softmax_fc(float *grad_fc, float *sfmax_input, float *answer, int mini_batch, int outsize)
{
    for (int i = 0; i < outsize * mini_batch; i++)
    {
        grad_fc[i] = sfmax_input[i] - answer[i];
    }
}

// Relu層への勾配の計算
void backward_relu(float *grad_relu, float *grad_fc, float *fc_input, float *grad_mx, float *grad_v, int mini_batch, float *matrix, int insize, int outsize)
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
            float sum = 0;
            for (int j = 0; j < outsize; j++)
            {
                sum += matrix[j * insize + n] * grad_fc[i * outsize + j];
            }
            grad_relu[i * insize + n] = sum;
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
void backward_pass(int i, float *train_x, float *answer, float *sfmax_input, float *final_output, float *fc3_input, float *relu2_input, float *fc2_input, float *relu1_input, float *matrix1, float *matrix2, float *matrix3, float *grad_fc3, float *grad_fc2, float *grad_fc1, float *grad_relu2, float *grad_relu1, float *grad_relu0, float *grad_mx3, float *grad_v3, float *grad_mx2, float *grad_v2, float *grad_mx1, float *grad_v1, int mini_batch)
{
    backward_softmax_fc(grad_fc3, sfmax_input, answer, mini_batch, out3);

    backward_relu(grad_relu2, grad_fc3, fc3_input, grad_mx3, grad_v3, mini_batch, matrix3, in3, out3);

    backward_fc(grad_fc2, grad_relu2, relu2_input, mini_batch, out2);

    backward_relu(grad_relu1, grad_fc2, fc2_input, grad_mx2, grad_v2, mini_batch, matrix2, in2, out2);

    backward_fc(grad_fc1, grad_relu1, relu1_input, mini_batch, out1);

    backward_relu(grad_relu0, grad_fc1, train_x, grad_mx1, grad_v1, mini_batch, matrix1, in1, out1);
}

// 行列やベクトルの勾配を初期化
void reset_grads(float *grad_mx1, float *grad_v1, float *grad_mx2, float *grad_v2, float *grad_mx3, float *grad_v3)
{
    for (int i = 0; i < in1 * out1; i++)
        grad_mx1[i] = 0;
    for (int i = 0; i < out1; i++)
        grad_v1[i] = 0;

    for (int i = 0; i < in2 * out2; i++)
        grad_mx2[i] = 0;
    for (int i = 0; i < out2; i++)
        grad_v2[i] = 0;

    for (int i = 0; i < in3 * out3; i++)
        grad_mx3[i] = 0;
    for (int i = 0; i < out3; i++)
        grad_v3[i] = 0;
}

// パラメータを更新
void update_parameters(float *matrix, float *vector, float *grad_mx, float *grad_v, int insize, int outsize, float learning_rate)
{
    for (int i = 0; i < insize * outsize; i++)
    {
        matrix[i] -= learning_rate * grad_mx[i];
    }

    for (int i = 0; i < outsize; i++)
    {
        vector[i] -= learning_rate * grad_v[i];
    }
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
    if (!fp)
    {
        perror("fopen");
        return;
    }

    // 行列 A の読み込み
    fread(matrix, sizeof(float), out_size * in_size, fp);

    // バイアス b の読み込み
    fread(vector, sizeof(float), out_size, fp);

    fclose(fp);
}

// 損失関数のミニバッチごとの平均を導出
float calc_loss(float *output, float *answer, int out3, int mini_batch)
{
    float loss = 0.0f;
    for (int j = 0; j < mini_batch; j++)
    {
        for (int i = 0; i < out3; i++)
        {
            loss -= answer[i + j * out3] * logf(output[i + j * out3] + 1e-7f);
        }
    }

    return loss / mini_batch;
}

// 出力からどの数字がどれくらいの確率で正解なのかを導出する関数
void correct_prediction(float *output, int out3, int *answer_num, float *max_prob)
{
    *answer_num = 0;
    *max_prob = output[0];
    for (int j = 1; j < out3; j++)
    {
        if (output[j] > *max_prob)
        {
            *max_prob = output[j];
            *answer_num = j;
        }
    }
}

// 結果を出力(解析した文字と正解の文字、テストの正答率)
void print_result(float *test_output, char *test_y, int test_count, int out3)
{
    float accuracy = 0.0f;

    for (int i = 0; i < test_count; i++)
    {
        float *output = &test_output[i * out3]; // 各テスト画像の出力部分だけ渡す
        int answer_num;
        float max_prob;

        correct_prediction(output, out3, &answer_num, &max_prob);

        int label = (int)test_y[i];

        if (answer_num == label)
        {
            accuracy += 1.0f;
        }
        else
        {
            printf("誤った予測結果 : %d (確率: %.2f%%), 正解 : %d\n", answer_num, max_prob * 100.0f, label);
        }
    }

    printf("正答率 : %.2f%%\n", (accuracy / test_count) * 100.0f);
}

// コンパイル後に実行する際の引数の数によって学習モードか判定モードかを分ける
int main(int argc, char *argv[])
{
    if (argc == 5)
    {
        float matrix1[in1 * out1];
        float vector1[out1];
        float matrix2[in2 * out2];
        float vector2[out2];
        float matrix3[in3 * out3];
        float vector3[out3];

        float *input = load_mnist_bmp(argv[4]);
        float output[out3];

        load_parameters(argv[1], out1, in1, matrix1, vector1);
        load_parameters(argv[2], out2, in2, matrix2, vector2);
        load_parameters(argv[3], out3, in3, matrix3, vector3);

        forward_pass_test(input, output, matrix1, vector1, matrix2, vector2, matrix3, vector3, 1);

        int answer_num = 0;
        float max_prob = output[0];

        for (int i = 0; i < 10; i++)
        {
            printf("%d : %f\n", i, output[i]);
        }

        correct_prediction(output, 10, &answer_num, &max_prob);

        printf("予測結果 : %d\n", answer_num);

        free(input);
    }
    else
    {
        srand((unsigned int)time(NULL)); // 乱数初期化

        float *train_x = NULL;
        unsigned char *train_y = NULL;
        int train_count = -1;

        float *test_x = NULL;
        unsigned char *test_y = NULL;
        int test_count = -1;

        int width = -1;
        int height = -1;

        load_mnist(&train_x, &train_y, &train_count,
                   &test_x, &test_y, &test_count,
                   &width, &height);

        // 行列とベクトルの配列をmallocでヒープ領域に確保
        float *matrix1 = malloc(sizeof(float) * in1 * out1);
        float *vector1 = malloc(sizeof(float) * out1);
        float *matrix2 = malloc(sizeof(float) * in2 * out2);
        float *vector2 = malloc(sizeof(float) * out2);
        float *matrix3 = malloc(sizeof(float) * in3 * out3);
        float *vector3 = malloc(sizeof(float) * out3);

        reset_parameters(matrix1, vector1, matrix2, vector2, matrix3, vector3);

        // 各層の入力をmallocでヒープ領域に確保
        float *relu1_input = malloc(sizeof(float) * mini_batch * out1);
        float *fc2_input = malloc(sizeof(float) * mini_batch * out1);
        float *relu2_input = malloc(sizeof(float) * mini_batch * out2);
        float *fc3_input = malloc(sizeof(float) * mini_batch * out2);
        float *sfmax_input = malloc(sizeof(float) * mini_batch * out3);
        float *final_output = malloc(sizeof(float) * mini_batch * out3);
        float *answer = malloc(sizeof(float) * mini_batch * out3);

        // 各関数の勾配はミニバッチ分確保し、行列とベクトルは平均を代入
        float *grad_fc3 = malloc(sizeof(float) * mini_batch * out3);
        float *grad_fc2 = malloc(sizeof(float) * mini_batch * out2);
        float *grad_fc1 = malloc(sizeof(float) * mini_batch * out1);
        float *grad_relu2 = malloc(sizeof(float) * mini_batch * out2);
        float *grad_relu1 = malloc(sizeof(float) * mini_batch * out1);
        float *grad_relu0 = malloc(sizeof(float) * mini_batch * in1);

        // 行列やベクトルの勾配をヒープ領域に確保
        float *grad_mx1 = calloc(in1 * out1, sizeof(float));
        float *grad_v1 = calloc(out1, sizeof(float));
        float *grad_mx2 = calloc(in2 * out2, sizeof(float));
        float *grad_v2 = calloc(out2, sizeof(float));
        float *grad_mx3 = calloc(in3 * out3, sizeof(float));
        float *grad_v3 = calloc(out3, sizeof(float));

        for (int j = 0; j < epoc; j++)
        {
            shuffle_train_data(train_x, (char *)train_y, train_count);

            for (int i = 0; i < train_count / mini_batch; i++)
            {
                reset_grads(grad_mx1, grad_v1, grad_mx2, grad_v2, grad_mx3, grad_v3);

                forward_pass(&train_x[i * in1 * mini_batch], final_output, matrix1, vector1, matrix2, vector2, matrix3, vector3, relu1_input, fc2_input, relu2_input, fc3_input, sfmax_input, mini_batch);

                set_answer(answer, (char *)&train_y[i * mini_batch], mini_batch, out3);

                backward_pass(i, &train_x[i * in1 * mini_batch], answer, sfmax_input, final_output, fc3_input, relu2_input, fc2_input, relu1_input, matrix1, matrix2, matrix3, grad_fc3, grad_fc2, grad_fc1, grad_relu2, grad_relu1, grad_relu0, grad_mx3, grad_v3, grad_mx2, grad_v2, grad_mx1, grad_v1, mini_batch);

                update_parameters(matrix1, vector1, grad_mx1, grad_v1, in1, out1, learning_rate);
                update_parameters(matrix2, vector2, grad_mx2, grad_v2, in2, out2, learning_rate);
                update_parameters(matrix3, vector3, grad_mx3, grad_v3, in3, out3, learning_rate);
            }

            printf("損失関数の平均 : %f\n", calc_loss(final_output, answer, out3, mini_batch));
        }

        float *params_2[] = {final_output, answer, grad_fc3, grad_fc2, grad_fc1, grad_relu2, grad_relu1, grad_relu0, relu1_input, fc2_input, relu2_input, fc3_input, sfmax_input, grad_mx1, grad_v1, grad_mx2, grad_v2, grad_mx3, grad_v3};
        free_many(params_2, sizeof(params_2) / sizeof(params_2[0]));

        save_parameters("parameter_fc1.bin", out1, in1, matrix1, vector1);
        save_parameters("parameter_fc2.bin", out2, in2, matrix2, vector2);
        save_parameters("parameter_fc3.bin", out3, in3, matrix3, vector3);

        float *test_output = malloc(sizeof(float) * out3 * test_count);
        forward_pass_test(test_x, test_output, matrix1, vector1, matrix2, vector2, matrix3, vector3, test_count);

        print_result(test_output, (char *)test_y, test_count, out3);

        float *params_3[] = {test_x, matrix1, vector1, matrix2, vector2, matrix3, vector3, test_output};
        free_many(params_3, sizeof(params_3) / sizeof(params_3[0]));
        free(test_y);
    }

/* 浮動小数点例外で停止することを確認するためのコード */
#if 0
    volatile float x = 0;
    volatile float y = 0;
    volatile float z = x/y;
#endif

    return 0;
}

/*
magick input.png -resize 28x28! -type Grayscale output.bmp
./final_assignment parameter_fc1.bin parameter_fc2.bin parameter_fc3.bin output.bmp ←pngを28*28に変換
*/
/*
// コンパイルして実行
clang final_assignment.c -o final_assignment
./final_assignment
*/
/*
//
magick -size 28x28 xc:black -pointsize 24 -fill white -draw "text 7,23 '4'" img4.bmp
./final_assignment parameter_fc1.bin parameter_fc2.bin parameter_fc3.bin img4.bmp
*/