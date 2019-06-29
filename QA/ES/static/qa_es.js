var app = angular.module('myApp1', []);
        app.controller('formCtrl', function ($scope, $http, $sce) {
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

                        });
                    dialogue.scrollTop=dialogue.scrollHeight;
                    // console.log($scope);
                }
            }
        });