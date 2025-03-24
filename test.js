import http from 'k6/http';
import { sleep } from "k6";

const data = JSON.parse(open("./sampleTorch.json"))

export const options = {
    vus: 25,
    duration: "2m",
};

export default function () {
    const getRandom = (array) => {
        const random = Math.floor(Math.random() * array.length);
        return array[random];
    }

    const url = 'http://localhost:8080/predictions/emotions';
  
    const res = http.post(url, JSON.stringify(getRandom(data)), {
      headers: { 'Content-Type': 'application/json' },
    });

    sleep(1)
}