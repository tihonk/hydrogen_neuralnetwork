import numpy as np
import torch

from http.server import BaseHTTPRequestHandler, HTTPServer

hostName = "localhost"
serverPort = 9102
model = torch.load('model.pth')


class MyServer(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(bytes("<body>", "utf-8"))
        self.wfile.write(bytes("<p>Python server works correctly.</p>", "utf-8"))
        self.wfile.write(bytes("</body></html>", "utf-8"))

    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        content = self.headers['body']
        matrix = np.array(content.split(',')[0:-1], dtype=np.float32)
        data_x = torch.tensor(matrix).to(torch.float32)
        output = model(data_x)
        self.wfile.write((str(output[0].item()) + ' ' + str(output[1].item()) + ' ' + str(output[2].item()) + ' ').encode(encoding='utf_8'))






def main():
    # n_input, n_hidden, n_out, batch_size, learning_rate = 78, 40, 3, 3, 0.01
    matrix = np.loadtxt('dataset/tr_set4.txt')
    data_x = torch.tensor(matrix[:, 0:-3]).to(torch.float32)
    # data_y = torch.tensor(matrix[:, -3:]).to(torch.float32)
    #
    # matrix_test = np.loadtxt('dataset/tst_set4.txt')
    # test_x = torch.tensor(matrix_test[:, 0:-3]).to(torch.float32)
    # test_y = torch.tensor(matrix_test[:, -3:]).to(torch.float32)

    # model = nn.Sequential(nn.Linear(n_input, n_hidden),
    #                       nn.ReLU(),
    #                       nn.Linear(n_hidden, n_hidden),
    #                       nn.ReLU(),
    #                       nn.Linear(n_hidden, n_hidden),
    #                       nn.ReLU(),
    #                       nn.Linear(n_hidden, n_hidden),
    #                       nn.ReLU(),
    #                       nn.Linear(n_hidden, n_hidden),
    #                       nn.ReLU(),
    #                       nn.Linear(n_hidden, n_out),
    #                       nn.Tanh())
    #
    # loss_function = nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    #
    # losses = []
    # losses_test = []
    # print('Loading...')
    #
    # pred_test_y = []
    #
    # for epoch in range(2000):
    #     pred_y = model(data_x)
    #     loss = loss_function(pred_y, data_y)
    #     losses.append(loss.item())
    #
    #     pred_test_y = model(test_x)
    #     loss_test = loss_function(pred_test_y, test_y)
    #     losses_test.append(loss_test.item())
    #
    #     model.zero_grad()
    #     loss.backward()
    #
    #     optimizer.step()


    # plt.plot(losses)
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.title("Learning rate %f" % (learning_rate))
    # plt.show()


if __name__ == '__main__':
    webServer = HTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")
    main()
