import { Injectable } from '@angular/core';
import { environment } from '../environments/environment';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { Token } from './interfaces/token';

@Injectable({
  providedIn: 'root'
})
export class ProofReadService {
  private readonly apiUrl : string = environment.apiUrl + "/proof_read"
  httpClient: HttpClient;

  constructor(httpClient: HttpClient) {
    this.httpClient = httpClient
   }

  getProofRead(sentence: string): Observable<Array<Token>>{
    const formData: FormData = new FormData()
    formData.append("sentence", sentence)
    return this.httpClient.post(this.apiUrl, formData) as Observable<Array<Token>>;
  }
}
