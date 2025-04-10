export interface Token {
    position: number,
    token: string,
    score: number,
    probability: number,
    top_alternatives: Array<any>
}