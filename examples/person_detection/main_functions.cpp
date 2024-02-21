/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "main_functions.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
extern "C" {
#include "port_common.h"
#include "w5x00_spi.h"
#include "wizchip_conf.h"
#include "loopback.h"
#include "socket.h"
}
#include "detection_responder.h"
#include "image_provider.h"
#include "model_settings.h"
#include "person_detect_model_data.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "arducam.h"
#include "pico/stdlib.h"
#include "hardware/irq.h"
#include "tensorflow/lite/micro/micro_time.h"
#include <climits>



#define TF_LITE_MICRO_EXECUTION_TIME_BEGIN      \
  int32_t start_ticks;                          \
  int32_t duration_ticks;                       \
  int32_t duration_ms;
#define TF_LITE_MICRO_EXECUTION_TIME(reporter, func)                    \
  if (tflite::ticks_per_second() == 0) {                                \
    TF_LITE_REPORT_ERROR(reporter,                                      \
                         "no timer implementation found");              \
  }                                                                     \
  start_ticks = tflite::GetCurrentTimeTicks();                          \
  func;                                                                 \
  duration_ticks = tflite::GetCurrentTimeTicks() - start_ticks;         \
  if (duration_ticks > INT_MAX / 1000) {                                \
    duration_ms = duration_ticks / (tflite::ticks_per_second() / 1000); \
  } else {                                                              \
    duration_ms = (duration_ticks * 1000) / tflite::ticks_per_second(); \
  }                                                                     \
  TF_LITE_REPORT_ERROR(reporter, "%s took %d ticks (%d ms)", #func,     \
                                    duration_ticks, duration_ms);
#define TF_LITE_MICRO_EXECUTION_TIME_SNIPPET_START(reporter)            \
  if (tflite::ticks_per_second() == 0) {                                \
    TF_LITE_REPORT_ERROR(reporter,                                      \
                         "no timer implementation found");              \
  }                                                                     \
  start_ticks = tflite::GetCurrentTimeTicks();

#define TF_LITE_MICRO_EXECUTION_TIME_SNIPPET_END(reporter, desc)        \
  duration_ticks = tflite::GetCurrentTimeTicks() - start_ticks;         \
  if (duration_ticks > INT_MAX / 1000) {                                \
    duration_ms = duration_ticks / (tflite::ticks_per_second() / 1000); \
  } else {                                                              \
    duration_ms = (duration_ticks * 1000) / tflite::ticks_per_second(); \
  }                                                                     \
  TF_LITE_REPORT_ERROR(reporter, "%s took %d ticks (%d ms)", desc,      \
                                    duration_ticks, duration_ms);

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

// In order to use optimized tensorflow lite kernels, a signed int8_t quantized
// model is preferred over the legacy unsigned model format. This means that
// throughout this project, input images must be converted from unisgned to
// signed format. The easiest and quickest way to convert from unsigned to
// signed 8-bit integers is to subtract 128 from the unsigned value to get a
// signed value.

// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 136 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

#ifndef DO_NOT_OUTPUT_TO_UART
// RX interrupt handler
void on_uart_rx() {
    uint8_t cameraCommand = 0;
    while (uart_is_readable(UART_ID)) {
        cameraCommand = uart_getc(UART_ID);
        // Can we send it back?
        if (uart_is_writable(UART_ID)) {
            uart_putc(UART_ID, cameraCommand);
        }
    }
}

void setup_uart() {
  // Set up our UART with the required speed.
  uint baud = uart_init(UART_ID, BAUD_RATE);
  // Set the TX and RX pins by using the function select on the GPIO
  // Set datasheet for more information on function select
  gpio_set_function(UART_TX_PIN, GPIO_FUNC_UART);
  gpio_set_function(UART_RX_PIN, GPIO_FUNC_UART);
  // Set our data format
  uart_set_format(UART_ID, DATA_BITS, STOP_BITS, PARITY);
  // Turn off FIFO's - we want to do this character by character
  uart_set_fifo_enabled(UART_ID, false);
  // Set up a RX interrupt
  // We need to set up the handler first
  // Select correct interrupt for the UART we are using
  int UART_IRQ = UART_ID == uart0 ? UART0_IRQ : UART1_IRQ;

  // And set up and enable the interrupt handlers
  irq_set_exclusive_handler(UART_IRQ, on_uart_rx);
  irq_set_enabled(UART_IRQ, true);

  // Now enable the UART to send interrupts - RX only
  uart_set_irq_enables(UART_ID, true, false);
}
#else
void setup_uart() {}
#endif

/* Clock */
#define PLL_SYS_KHZ (133 * 1000)

/* Buffer */
#define ETHERNET_BUF_MAX_SIZE (1024 * 2)

/* Socket */
#define ETHERNET_SOCKET 0

extern char __StackLimit, __end__;
#define HEAPSZ (&__StackLimit - &__end__)

/* Network */
static wiz_NetInfo g_net_info =
    {
        .mac = {0x00, 0x08, 0xDC, 0x12, 0x34, 0x56}, // MAC address
        .ip = {192, 168, 11, 28},                     // IP address
        .sn = {255, 255, 255, 0},                    // Subnet Mask
        .gw = {192, 168, 11, 1},                     // Gateway
        .dns = {8, 8, 8, 8},                         // DNS server
        .dhcp = NETINFO_STATIC                       // DHCP enable/disable
};

static uint8_t des_ip[4] = {192, 168, 11, 25}; // Server IP address
static uint16_t des_port = 5000;             // Server port

/* Loopback */
static uint8_t g_loopback_buf[ETHERNET_BUF_MAX_SIZE] = {
    0,
};


/* Clock */
static void set_clock_khz(void)
{
    // set a system clock frequency in khz
    set_sys_clock_khz(PLL_SYS_KHZ, true);

    // configure the specified clock
    clock_configure(
        clk_peri,
        0,                                                // No glitchless mux
        CLOCKS_CLK_PERI_CTRL_AUXSRC_VALUE_CLKSRC_PLL_SYS, // System PLL on AUX mux
        PLL_SYS_KHZ * 1000,                               // Input frequency
        PLL_SYS_KHZ * 1000                                // Output (must be same as no divider)
    );
}

void setup_wiznet() {

  int retval = 0;


  wizchip_spi_initialize(PLL_SYS_KHZ / 4 * 1000);
  wizchip_cris_initialize();
  wizchip_reset();
  wizchip_initialize();
  wizchip_check();
  network_initialize(g_net_info);
  /* Get network information */
  print_network_information(g_net_info);

}

// The name of this function is important for Arduino compatibility.
void setup() {
  setup_uart();
  setup_wiznet();
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver<5> micro_op_resolver;
  micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax();

  // Build an interpreter to run the model with.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);
}

// The name of this function is important for Arduino compatibility.
void loop() {
  TF_LITE_MICRO_EXECUTION_TIME_BEGIN

  TF_LITE_MICRO_EXECUTION_TIME_SNIPPET_START(error_reporter)
  
  loopback_tcpc(ETHERNET_SOCKET, g_loopback_buf, des_ip, des_port);
  // Get image from provider.
  if (kTfLiteOk != GetImage(error_reporter, kNumCols, kNumRows, kNumChannels,
                            input->data.int8)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Image capture failed.");
  }
  TF_LITE_MICRO_EXECUTION_TIME_SNIPPET_END(error_reporter, "GetImage")

  TF_LITE_MICRO_EXECUTION_TIME_SNIPPET_START(error_reporter)
  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke()) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
  }
  TF_LITE_MICRO_EXECUTION_TIME_SNIPPET_END(error_reporter, "Invoke")

  TfLiteTensor* output = interpreter->output(0);

  // Process the inference results.
  int8_t person_score = output->data.uint8[kPersonIndex];
  int8_t no_person_score = output->data.uint8[kNotAPersonIndex];
  
  char dataBuf[100] = {0};
  sprintf(dataBuf, "Person: %.2f%%\r\n", ((person_score + 128)/256.0 * 100));
  
  if (getSn_SR(ETHERNET_SOCKET) == SOCK_ESTABLISHED){
    send(ETHERNET_SOCKET, (uint8_t*) dataBuf, 50);
  } else {
    printf("Socket Disconnected\r\n");
  }


  RespondToDetection(error_reporter, person_score, no_person_score);
}
