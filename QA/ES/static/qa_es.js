var app = angular.module('myApp1', []);
        app.controller('formCtrl', function ($scope, $http, $sce) {
            // 设置默认回答
            setTimeout(
                () => {var dialogue = document.getElementById('dialogue');
                dialogue.innerHTML = '<li><span class="spanleft">请输入一句话开启客服助手：<br/>例子：<br/>我要洗相片<br/>我想做文化衫<br/>今天去哪儿玩</span></li>';},
                500
            );

            $scope.getResponse = function () {
                var mydata = $scope.text;
                var dialogue = document.getElementById('dialogue');
                var result = document.getElementById('result');
                // console.log(mydata);
                if (mydata == '' || typeof(mydata) == "undefined") {
                    alert('不能发送空消息');
                    return;
                }else {
                    results = "";
                    $http.post('/api/getResponse/', mydata)
                        .success(function (data) {
                            results += '<li><span class="spanright">' + mydata + '</span></li>';
                            results += '<li><span class="spanleft">' + data['result'][0]['answer'] + '</span></li>';
                            console.log(results);
                            dialogue.innerHTML += results;

                            result.innerHTML = "";
                            // result;
                            // $scope.dialogue += $sce.trustAsHtml(results);
                            //
                            setTimeout(() => {
                                 dialogue.scrollTop=dialogue.scrollHeight;
                            }, 500)

                        });
                    dialogue.scrollTop=dialogue.scrollHeight;
                    // console.log($scope);
                }
            }
        });