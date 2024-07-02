import{_ as s,c as a,o as n,a3 as i}from"./chunks/framework.CRN3gama.js";const y=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),p={name:"tutorials/beginner/3_SimpleRNN.md"},l=i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxAMDGPU, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">      Statistics</span></span></code></pre></div><h2 id="Dataset" tabindex="-1">Dataset <a class="header-anchor" href="#Dataset" aria-label="Permalink to &quot;Dataset {#Dataset}&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create the spirals</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [MLUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Datasets</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">make_spiral</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(sequence_length) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> _ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">dataset_size]</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Get the labels</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    labels </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">repeat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">repeat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    clockwise_spirals </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(d[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">][:, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">sequence_length], :, sequence_length, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">                         for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> d </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)]]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    anticlockwise_spirals </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                                 d[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">][:, (sequence_length </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], :, sequence_length, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">                             for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> d </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data[((dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Float32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">cat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(clockwise_spirals</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, anticlockwise_spirals</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; dims</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Split the dataset</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (x_train, y_train), (x_val, y_val) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> splitobs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x_data, labels); at</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create DataLoaders</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Use DataLoader to automatically minibatch and shuffle the data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_train, y_train)); batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Don&#39;t shuffle the validation data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_val, y_val)); batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractExplicitContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/v0.5.47/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">:</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">       Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">AbstractExplicitContainerLayer{(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/v0.5.47/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/v0.5.47/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>Main.var&quot;##225&quot;.SpiralClassifier</span></span></code></pre></div><p>We can use default Lux blocks – <code>Recurrence(LSTMCell(in_dims =&gt; hidden_dims)</code> – instead of defining the following. But let&#39;s still do it for the sake of it.</p><p>Now we need to define the behavior of the Classifier when it is invoked.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T, 3}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # First we will have to run the sequence through the LSTM Cell</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # The first call to LSTM Cell will create the initial hidden state</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # See that the parameters and states are automatically populated into a field called</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # \`lstm_cell\` We use \`eachslice\` to get the elements in the sequence without copying,</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # and \`Iterators.peel\` to split out the first element for LSTM initialization.</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_init, x_rest </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Iterators</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">peel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">_eachslice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (y, carry), st_lstm </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_init, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lstm_cell, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lstm_cell)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Now that we have the hidden state and memory in \`carry\` we will pass the input and</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # \`carry\` jointly</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x_rest</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        (y, carry), st_lstm </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, carry), ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lstm_cell, st_lstm)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # After running through the sequence we will pass the output through the classifier</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y, st_classifier </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">classifier, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">classifier)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Finally remember to create the updated state</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> merge</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st, (classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">st_classifier, lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">st_lstm))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vec</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y), st</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/v0.5.47/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; lstm_cell, classifier) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T, 3}</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        x_init, x_rest </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Iterators</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">peel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">_eachslice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        y, carry </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_init)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x_rest</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y, carry </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, carry))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vec</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>SpiralClassifierCompact (generic function with 1 method)</span></span></code></pre></div><h2 id="Defining-Accuracy,-Loss-and-Optimiser" tabindex="-1">Defining Accuracy, Loss and Optimiser <a class="header-anchor" href="#Defining-Accuracy,-Loss-and-Optimiser" aria-label="Permalink to &quot;Defining Accuracy, Loss and Optimiser {#Defining-Accuracy,-Loss-and-Optimiser}&quot;">​</a></h2><p>Now let&#39;s define the binarycrossentropy loss. Typically it is recommended to use <code>logitbinarycrossentropy</code> since it is more numerically stable, but for the sake of simplicity we will use <code>binarycrossentropy</code>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> xlogy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, y)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    result </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> log</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> ifelse</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">iszero</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">zero</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(result), result)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> binarycrossentropy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_pred </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y_pred </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> eps</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">eltype</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> mean</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@.</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> -</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">xlogy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_true, y_pred) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> xlogy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> -</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y_true, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> -</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y_pred))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> compute_loss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, (x, y))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_pred, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, ps, st)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> binarycrossentropy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y), st, (; y_pred</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">y_pred)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">matches</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> sum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((y_pred </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.5f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.==</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y_true)</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> matches</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>accuracy (generic function with 1 method)</span></span></code></pre></div><h2 id="Training-the-Model" tabindex="-1">Training the Model <a class="header-anchor" href="#Training-the-Model" aria-label="Permalink to &quot;Training the Model {#Training-the-Model}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model_type)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Get the dataloaders</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (train_loader, val_loader) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model_type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    rng </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Xoshiro</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Experimental</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TrainState</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        rng, model, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Adam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.01f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">); transform_variables</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">dev)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">25</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Train the model</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> train_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            gs, loss, _, train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Experimental</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">compute_gradients</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                AutoZygote</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), compute_loss, (x, y), train_state)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Experimental</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">apply_gradients</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state, gs)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Epoch [%3d]: Loss %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch loss</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Validate the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> val_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            loss, st_, ret </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> compute_loss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, st_, (x, y))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ret</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">y_pred, y)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Validation: Loss %4.5f Accuracy %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss acc</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>Epoch [  1]: Loss 0.56378</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50752</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48376</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.44901</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.41898</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40592</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.37082</span></span>
<span class="line"><span>Validation: Loss 0.36514 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37179 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36330</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35605</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.33635</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31941</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.29596</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28791</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.26917</span></span>
<span class="line"><span>Validation: Loss 0.25600 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.26055 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25700</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24355</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23244</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22163</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20889</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19993</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19140</span></span>
<span class="line"><span>Validation: Loss 0.17843 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.18094 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17869</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17343</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16175</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15648</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14601</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13936</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13323</span></span>
<span class="line"><span>Validation: Loss 0.12691 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12838 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12780</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11958</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11765</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11305</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10422</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10171</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09577</span></span>
<span class="line"><span>Validation: Loss 0.09192 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09322 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09217</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08940</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08558</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08106</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07689</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07327</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07194</span></span>
<span class="line"><span>Validation: Loss 0.06755 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06881 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06801</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06586</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06222</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05916</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05907</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05438</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05359</span></span>
<span class="line"><span>Validation: Loss 0.05023 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05143 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05154</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04825</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04827</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04437</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04190</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04187</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.03854</span></span>
<span class="line"><span>Validation: Loss 0.03770 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03882 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03754</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03636</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03522</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03429</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03347</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03264</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.02648</span></span>
<span class="line"><span>Validation: Loss 0.02875 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02980 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02729</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02945</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02762</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02536</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02723</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02392</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02436</span></span>
<span class="line"><span>Validation: Loss 0.02259 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02355 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02304</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02319</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02135</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02145</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01917</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02018</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01926</span></span>
<span class="line"><span>Validation: Loss 0.01838 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01921 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01987</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01766</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01693</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01699</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01669</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01735</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01700</span></span>
<span class="line"><span>Validation: Loss 0.01545 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01617 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01576</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01481</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01584</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01524</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01435</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01380</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01393</span></span>
<span class="line"><span>Validation: Loss 0.01334 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01398 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01306</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01343</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01305</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01345</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01268</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01224</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01325</span></span>
<span class="line"><span>Validation: Loss 0.01176 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01233 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01175</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01149</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01242</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01193</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01150</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01042</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01033</span></span>
<span class="line"><span>Validation: Loss 0.01053 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01103 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01044</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01040</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01081</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01056</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01050</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00948</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01043</span></span>
<span class="line"><span>Validation: Loss 0.00953 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00999 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00980</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00959</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00925</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00962</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00878</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00931</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01003</span></span>
<span class="line"><span>Validation: Loss 0.00870 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00912 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00925</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00868</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00855</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00891</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00805</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00848</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00778</span></span>
<span class="line"><span>Validation: Loss 0.00799 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00838 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00807</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00828</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00793</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00778</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00795</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00785</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00695</span></span>
<span class="line"><span>Validation: Loss 0.00738 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00775 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00729</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00802</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00689</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00752</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00708</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00732</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00727</span></span>
<span class="line"><span>Validation: Loss 0.00685 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00720 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00730</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00719</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00647</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00681</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00687</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00681</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00527</span></span>
<span class="line"><span>Validation: Loss 0.00639 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00672 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00669</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00681</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00655</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00635</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00602</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00605</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00582</span></span>
<span class="line"><span>Validation: Loss 0.00598 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00629 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00648</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00602</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00602</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00582</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00577</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00576</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00616</span></span>
<span class="line"><span>Validation: Loss 0.00561 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00591 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00574</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00587</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00536</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00565</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00538</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00566</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00596</span></span>
<span class="line"><span>Validation: Loss 0.00529 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00556 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00539</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00529</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00496</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00519</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00551</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00531</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00591</span></span>
<span class="line"><span>Validation: Loss 0.00499 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00525 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>Epoch [  1]: Loss 0.56239</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51236</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47175</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.44943</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.41743</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40377</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.37358</span></span>
<span class="line"><span>Validation: Loss 0.36521 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36996 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37198</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.34898</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.32412</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31548</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30156</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28214</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.27764</span></span>
<span class="line"><span>Validation: Loss 0.25536 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.25853 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25110</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24377</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23213</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21916</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20794</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19779</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19007</span></span>
<span class="line"><span>Validation: Loss 0.17735 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17905 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.18035</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16594</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16056</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15579</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14579</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13752</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13215</span></span>
<span class="line"><span>Validation: Loss 0.12575 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12682 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12645</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11986</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11516</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11071</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10436</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09839</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09791</span></span>
<span class="line"><span>Validation: Loss 0.09089 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09184 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09126</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08632</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08456</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08069</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07605</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07245</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.06996</span></span>
<span class="line"><span>Validation: Loss 0.06676 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06768 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06717</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06408</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06110</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06006</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05741</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05418</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.04729</span></span>
<span class="line"><span>Validation: Loss 0.04965 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05053 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04813</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04820</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04670</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04462</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04252</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04110</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.03837</span></span>
<span class="line"><span>Validation: Loss 0.03736 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03821 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03631</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03581</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03488</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03552</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03270</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03040</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.02793</span></span>
<span class="line"><span>Validation: Loss 0.02861 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02938 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02979</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02686</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02637</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02566</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02489</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02515</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02295</span></span>
<span class="line"><span>Validation: Loss 0.02256 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02325 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02335</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02146</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02158</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01980</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02067</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01961</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01958</span></span>
<span class="line"><span>Validation: Loss 0.01840 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01900 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01856</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01737</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01692</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01719</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01741</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01657</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01729</span></span>
<span class="line"><span>Validation: Loss 0.01548 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01600 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01615</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01494</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01482</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01423</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01489</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01344</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01489</span></span>
<span class="line"><span>Validation: Loss 0.01337 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01381 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01274</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01269</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01266</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01269</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01381</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01232</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01308</span></span>
<span class="line"><span>Validation: Loss 0.01180 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01219 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01165</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01244</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01118</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01174</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01083</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01079</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01049</span></span>
<span class="line"><span>Validation: Loss 0.01056 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01091 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01138</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01029</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01028</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00985</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01012</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00973</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00975</span></span>
<span class="line"><span>Validation: Loss 0.00956 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00988 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00972</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00972</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00907</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00942</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00914</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00878</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00945</span></span>
<span class="line"><span>Validation: Loss 0.00873 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00903 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00864</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00848</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00807</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00900</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00812</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00894</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00819</span></span>
<span class="line"><span>Validation: Loss 0.00803 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00830 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00825</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00792</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00758</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00802</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00758</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00778</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00799</span></span>
<span class="line"><span>Validation: Loss 0.00742 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00768 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00749</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00747</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00721</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00702</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00754</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00706</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00686</span></span>
<span class="line"><span>Validation: Loss 0.00689 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00714 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00705</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00676</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00642</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00705</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00706</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00643</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00617</span></span>
<span class="line"><span>Validation: Loss 0.00643 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00665 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00630</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00669</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00612</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00675</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00612</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00606</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00594</span></span>
<span class="line"><span>Validation: Loss 0.00602 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00623 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00611</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00606</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00575</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00594</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00582</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00601</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00539</span></span>
<span class="line"><span>Validation: Loss 0.00565 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00585 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00585</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00538</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00541</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00590</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00524</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00533</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00680</span></span>
<span class="line"><span>Validation: Loss 0.00532 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00551 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00557</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00527</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00531</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00548</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00493</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00505</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00482</span></span>
<span class="line"><span>Validation: Loss 0.00502 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00520 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {compress </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">} ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
<span class="line"><span> :ps_trained</span></span>
<span class="line"><span> :st_trained</span></span></code></pre></div><h2 id="Appendix" tabindex="-1">Appendix <a class="header-anchor" href="#Appendix" aria-label="Permalink to &quot;Appendix {#Appendix}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">InteractiveUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxCUDA) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> CUDA</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    CUDA</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxAMDGPU) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> LuxAMDGPU</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    AMDGPU</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>Julia Version 1.10.3</span></span>
<span class="line"><span>Commit 0b4590a5507 (2024-04-30 10:59 UTC)</span></span>
<span class="line"><span>Build Info:</span></span>
<span class="line"><span>  Official https://julialang.org/ release</span></span>
<span class="line"><span>Platform Info:</span></span>
<span class="line"><span>  OS: Linux (x86_64-linux-gnu)</span></span>
<span class="line"><span>  CPU: 48 × AMD EPYC 7402 24-Core Processor</span></span>
<span class="line"><span>  WORD_SIZE: 64</span></span>
<span class="line"><span>  LIBM: libopenlibm</span></span>
<span class="line"><span>  LLVM: libLLVM-15.0.7 (ORCJIT, znver2)</span></span>
<span class="line"><span>Threads: 8 default, 0 interactive, 4 GC (on 2 virtual cores)</span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>  JULIA_CPU_THREADS = 2</span></span>
<span class="line"><span>  JULIA_AMDGPU_LOGGING_ENABLED = true</span></span>
<span class="line"><span>  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6</span></span>
<span class="line"><span>  LD_LIBRARY_PATH = /usr/local/nvidia/lib:/usr/local/nvidia/lib64</span></span>
<span class="line"><span>  JULIA_NUM_THREADS = 8</span></span>
<span class="line"><span>  JULIA_CUDA_HARD_MEMORY_LIMIT = 100%</span></span>
<span class="line"><span>  JULIA_PKG_PRECOMPILE_AUTO = 0</span></span>
<span class="line"><span>  JULIA_DEBUG = Literate</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA runtime 12.4, artifact installation</span></span>
<span class="line"><span>CUDA driver 12.4</span></span>
<span class="line"><span>NVIDIA driver 550.54.15</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA libraries: </span></span>
<span class="line"><span>- CUBLAS: 12.4.5</span></span>
<span class="line"><span>- CURAND: 10.3.5</span></span>
<span class="line"><span>- CUFFT: 11.2.1</span></span>
<span class="line"><span>- CUSOLVER: 11.6.1</span></span>
<span class="line"><span>- CUSPARSE: 12.3.1</span></span>
<span class="line"><span>- CUPTI: 22.0.0</span></span>
<span class="line"><span>- NVML: 12.0.0+550.54.15</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Julia packages: </span></span>
<span class="line"><span>- CUDA: 5.3.3</span></span>
<span class="line"><span>- CUDA_Driver_jll: 0.8.1+0</span></span>
<span class="line"><span>- CUDA_Runtime_jll: 0.12.1+0</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Toolchain:</span></span>
<span class="line"><span>- Julia: 1.10.3</span></span>
<span class="line"><span>- LLVM: 15.0.7</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>- JULIA_CUDA_HARD_MEMORY_LIMIT: 100%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.297 GiB / 4.750 GiB available)</span></span>
<span class="line"><span>┌ Warning: LuxAMDGPU is loaded but the AMDGPU is not functional.</span></span>
<span class="line"><span>└ @ LuxAMDGPU ~/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/LuxAMDGPU/sGa0S/src/LuxAMDGPU.jl:19</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45),h=[l];function e(t,k,c,o,E,r){return n(),a("div",null,h)}const g=s(p,[["render",e]]);export{y as __pageData,g as default};
